#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_v7_1_multi.py
Multi-tâche & multi-modal :
- Direction (bearish/neutral/bullish) : classification 3 classes sur 30/60/120m
- Amplitude (retour signé) : régression continue sur 30/60/120m (HuberLoss)
- Fusion Texte (DeBERTa) + Features marché (feat_*)

Prérequis dataset :
- Fichier: events_with_features.csv (généré par make_event_labels.py + attach_market_features.py)
- Colonnes requises :
  * event_time (UTC ISO)
  * titles_joined, body_concat (texte)
  * label_30m, label_60m, label_120m ∈ {bearish, neutral, bullish}
  * ret_30m, ret_60m, ret_120m (rendements signés)
  * mag_30m, mag_60m, mag_120m (amplitude continue, |ret|)
  * feat_* (features marché à t0)

Sorties :
- models/bert_v7_1_plus/ (par défaut — overridable via $OUTPUT_DIR)
  * poids + tokenizer
  * best.json
  * training_log.txt, metrics_log.jsonl
  * thresholds_mag.json (seuils small/medium/large appris sur train)
  * reports direction & magnitude (classification via seuils)
"""

import os, json, math, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm
from log_utils import append_run_log, default_params, print_training_log

# =====================
# Config (éditable)
# =====================
DATA_FILE = "events_with_features.csv"
TEXT_COLS = ["titles_joined", "body_concat"]
TIME_COL = "event_time"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "models/bert_v7_1_plus")
SEED = 42
USE_AMP = True
EARLY_STOP_PATIENCE = 2

# Pertes / pondération
FOCAL_GAMMA = 2.0           # ↑ renforcé (était 1.5)
LAMBDA_DIR = 1.2            # ↑ priorité direction (était 1.0)
LAMBDA_RET = 0.6            # pondération de la perte de retour signé
HUBER_DELTA = 0.004         # delta Huber sur ret (à ajuster selon échelle de ret)

# Labels / classes
LABELS = ["bearish","neutral","bullish"]
NUM_LABELS = 3

RET_COLS = ["ret_30m", "ret_60m", "ret_120m"]
MAG_COLS = ["mag_30m", "mag_60m", "mag_120m"]
LABEL_TO_SIGN = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# AMP compat (torch.amp vs torch.cuda.amp)
# =====================
def _make_amp_tools():
    try:
        from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew
        def _autocast_ctx(enabled):
            return _autocast_new(device_type="cuda", enabled=(enabled and DEVICE=="cuda"))
        def _make_scaler(enabled):
            return _GradScalerNew("cuda", enabled=(enabled and DEVICE=="cuda"))
        return _autocast_ctx, _make_scaler
    except Exception:
        from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld
        def _autocast_ctx(enabled):
            return _autocast_old(enabled=(enabled and DEVICE=="cuda"))
        def _make_scaler(enabled):
            return _GradScalerOld(enabled=(enabled and DEVICE=="cuda"))
        return _autocast_ctx, _make_scaler

autocast_ctx, make_scaler = _make_amp_tools()

# =====================
# Utils
# =====================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =====================
# Data handling
# =====================
def concat_text(row: pd.Series) -> str:
    parts = [(str(row.get(c, "")) if pd.notna(row.get(c, "")) else "") for c in TEXT_COLS]
    return " ".join(p.strip() for p in parts if p and p.strip())

def temporal_split(df: pd.DataFrame, val_ratio=0.2, time_col=TIME_COL):
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        cut = int(len(df)*(1-val_ratio))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    # fallback random
    idx = np.arange(len(df)); np.random.shuffle(idx)
    cut = int(len(df)*(1-val_ratio))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()

def select_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("feat_")]

def fit_feature_norm(train_feats: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in train_feats.columns:
        m = float(np.nanmean(train_feats[c].values))
        s = float(np.nanstd(train_feats[c].values) + 1e-12)
        stats[c] = {"mean": m, "std": s}
    return stats

def apply_feature_norm(feats: pd.DataFrame, stats: Dict[str, Dict[str,float]]) -> pd.DataFrame:
    out = feats.copy()
    for c, d in stats.items():
        if c in out.columns:
            out[c] = (out[c] - d["mean"]) / d["std"]
    return out

def learn_mag_thresholds(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Seuils small/medium/large appris sur la partie train (anti-fuite)
    thresholds[h] = {"q1": x, "q2": y}
    """
    thresholds = {}
    for h in [30,60,120]:
        col_ret = f"ret_{h}m"
        if col_ret in train_df.columns:
            vals = np.abs(train_df[col_ret].dropna().values)
        else:
            vals = train_df[f"mag_{h}m"].dropna().values
        if len(vals) < 100:
            q1, q2 = 0.0025, 0.0070
        else:
            q1, q2 = np.quantile(vals, [1/3, 2/3])
        thresholds[str(h)] = {"q1": float(q1), "q2": float(q2)}
    return thresholds

def mag_to_class(x: float, q1: float, q2: float) -> int:
    # 0: small, 1: medium, 2: large
    if np.isnan(x): return -1
    if x < q1: return 0
    if x < q2: return 1
    return 2

class MultiModalDataset(Dataset):
    def __init__(self, texts_enc, feat_mat, y_dir, y_ret, y_abs):
        self.enc = texts_enc          # dict with input_ids, attention_mask (torch.Tensor)
        self.feat = feat_mat          # torch.FloatTensor [N, F]
        self.y_dir = y_dir            # dict horizon-> Long
        self.y_ret = y_ret            # dict horizon-> Float (ret signé)
        self.y_abs = y_abs            # dict horizon-> Float (|ret|)
    def __len__(self): return self.enc["input_ids"].shape[0]
    def __getitem__(self, idx):
        return (
            self.enc["input_ids"][idx],
            self.enc["attention_mask"][idx],
            self.feat[idx],
            self.y_dir[30][idx], self.y_dir[60][idx], self.y_dir[120][idx],
            self.y_ret[30][idx], self.y_ret[60][idx], self.y_ret[120][idx],
            self.y_abs[30][idx], self.y_abs[60][idx], self.y_abs[120][idx],
        )

# =====================
# Model
# =====================
class MultiModalHead(nn.Module):
    """
    Texte -> backbone -> cls_hidden (H)
    Features marché -> MLP -> feat_hidden (H)
    Fusion = concat puis Linear pour revenir à H
    Heads :
      - dir_30/60/120 : Linear(H, 3)
      - mag_30/60/120 : Linear(H, 1) (régression)
    """
    def __init__(self, model_name: str, feat_dim: int, hidden_drop=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        H = self.backbone.config.hidden_size
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, H),
            nn.GELU(),
            nn.Dropout(hidden_drop),
            nn.Linear(H, H),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(H + H, H),
            nn.GELU(),
            nn.Dropout(hidden_drop)
        )
        self.dir30 = nn.Linear(H, 3)
        self.dir60 = nn.Linear(H, 3)
        self.dir120 = nn.Linear(H, 3)
        self.mag30 = nn.Linear(H, 1)
        self.mag60 = nn.Linear(H, 1)
        self.mag120 = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask, feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]          # [B, H]
        f = self.feat_mlp(feats)                   # [B, H]
        z = self.fuse(torch.cat([cls, f], dim=-1)) # [B, H]

        o_dir30 = self.dir30(z)
        o_dir60 = self.dir60(z)
        o_dir120 = self.dir120(z)

        o_mag30 = self.mag30(z).squeeze(-1)
        o_mag60 = self.mag60(z).squeeze(-1)
        o_mag120 = self.mag120(z).squeeze(-1)
        return (o_dir30, o_dir60, o_dir120), (o_mag30, o_mag60, o_mag120)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):  # gamma ↑
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.gamma = gamma
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        if self.gamma <= 0:
            return ce
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        mod = (1 - pt).pow(self.gamma)
        return (mod * ce).mean()

# =====================
# Train / Eval
# =====================
def compute_class_weights(y: torch.Tensor, num_classes=3) -> torch.Tensor:
    cnt = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
    w = 1.0 / cnt
    return (w / w.mean()).to(DEVICE)

class EarlyStopping:
    def __init__(self, patience=2, mode="max"):
        self.p = patience; self.mode = mode
        self.best = -float("inf") if mode=="max" else float("inf")
        self.bad = 0; self.stop = False
    def step(self, val):
        imp = (val > self.best) if self.mode=="max" else (val < self.best)
        if imp:
            self.best = val; self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.p: self.stop = True
        return imp

@torch.no_grad()
def evaluate(model, loader, thresholds):
    model.eval()
    true_dir = {30: [], 60: [], 120: []}
    pred_dir = {30: [], 60: [], 120: []}
    true_ret = {30: [], 60: [], 120: []}
    pred_ret = {30: [], 60: [], 120: []}
    true_abs = {30: [], 60: [], 120: []}

    for ids, mask, feats, y30, y60, y120, r30, r60, r120, a30, a60, a120 in loader:
        ids = ids.to(DEVICE); mask = mask.to(DEVICE); feats = feats.to(DEVICE)
        (o30,o60,o120), (g30,g60,g120) = model(ids, mask, feats)
        p30 = o30.argmax(-1).cpu(); p60 = o60.argmax(-1).cpu(); p120 = o120.argmax(-1).cpu()
        true_dir[30].append(y30); true_dir[60].append(y60); true_dir[120].append(y120)
        pred_dir[30].append(p30); pred_dir[60].append(p60); pred_dir[120].append(p120)
        true_ret[30].append(r30); true_ret[60].append(r60); true_ret[120].append(r120)
        pred_ret[30].append(g30.cpu()); pred_ret[60].append(g60.cpu()); pred_ret[120].append(g120.cpu())
        true_abs[30].append(a30); true_abs[60].append(a60); true_abs[120].append(a120)

    res = {}
    for h in [30,60,120]:
        yt = torch.cat(true_dir[h]).numpy()
        yp = torch.cat(pred_dir[h]).numpy()
        acc = float(accuracy_score(yt, yp))
        f1  = float(f1_score(yt, yp, average="macro"))

        tm = torch.cat(true_ret[h]).numpy()
        pm = torch.cat(pred_ret[h]).numpy()
        mae = float(mean_absolute_error(tm, pm))

        true_abs_vals = torch.cat(true_abs[h]).numpy()
        pred_abs_vals = np.abs(pm)

        q1 = thresholds[str(h)]["q1"]; q2 = thresholds[str(h)]["q2"]
        tm_cls = np.array([0 if np.isnan(x) or x<q1 else (1 if x<q2 else 2) for x in true_abs_vals])
        pm_cls = np.array([0 if x<q1 else (1 if x<q2 else 2) for x in pred_abs_vals])
        # enlever NaN labels
        mask = tm_cls >= 0
        if mask.sum() > 0:
            acc_mag = float(accuracy_score(tm_cls[mask], pm_cls[mask]))
            f1_mag  = float(f1_score(tm_cls[mask], pm_cls[mask], average="macro"))
        else:
            acc_mag, f1_mag = float("nan"), float("nan")

        # corrélation (robustesse d'ordre)
        try:
            spr, _ = spearmanr(tm, pm, nan_policy="omit")
            spr = float(0.0 if np.isnan(spr) else spr)
        except Exception:
            spr = 0.0

        res[h] = {
            "dir_acc": acc,
            "dir_f1": f1,
            "ret_mae": mae,
            "mag_acc_cls": acc_mag,
            "mag_f1_cls": f1_mag,
            "ret_spearman": spr,
            "dir_y_true": yt.tolist(),
            "dir_y_pred": yp.tolist(),
            "ret_y_true": tm.tolist(),
            "ret_y_pred": pm.tolist(),
            "abs_y_true": true_abs_vals.tolist(),
            "abs_y_pred": pred_abs_vals.tolist(),
        }
    return res

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_log.jsonl")
    # reset metrics log
    if os.path.exists(metrics_path): os.remove(metrics_path)
    logf = open(log_path, "a", encoding="utf-8")

    print(f"[i] Device: {DEVICE}")
    print(f"[i] Output: {OUTPUT_DIR}")
    logf.write(f"[i] Device: {DEVICE}\n")

    # ----- Load data -----
    df = pd.read_csv(DATA_FILE)

    # Vérification colonnes directionnelles
    label_cols = [f"label_{h}m" for h in [30, 60, 120]]
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    for col in RET_COLS:
        if col not in df.columns:
            horizon = col.split("_")[1]
            mag_col = f"mag_{horizon}"
            label_col = f"label_{horizon}"
            if mag_col not in df.columns or label_col not in df.columns:
                raise ValueError(f"Unable to reconstruct {col}: missing {mag_col} or {label_col}")
            signs = df[label_col].map(LABEL_TO_SIGN).astype(float)
            df[col] = df[mag_col].astype(float) * signs

    for h in [30, 60, 120]:
        mag_col = f"mag_{h}m"
        ret_col = f"ret_{h}m"
        if mag_col not in df.columns:
            df[mag_col] = df[ret_col].abs()

    for c in label_cols:
        df = df[df[c].isin(LABELS)]

    df = df.dropna(subset=RET_COLS).reset_index(drop=True)
    for col in RET_COLS + MAG_COLS:
        df[col] = df[col].astype(float)

    # split temporel
    train_df, val_df = temporal_split(df, val_ratio=0.2, time_col=TIME_COL)

    # thresholds magnitude (small/medium/large) appris sur train
    thresholds = learn_mag_thresholds(train_df)
    with open(os.path.join(OUTPUT_DIR, "thresholds_mag.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # texte
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def enc_texts(df_):
        texts = [concat_text(r) for _, r in df_.iterrows()]
        return tok(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

    # features marché
    feat_cols = select_feature_cols(df)
    if len(feat_cols) == 0:
        raise ValueError("Aucune colonne feat_* trouvée. Exécute attach_market_features.py au préalable.")
    train_feats_raw = train_df[feat_cols].astype(float)
    feat_norm = fit_feature_norm(train_feats_raw)
    with open(os.path.join(OUTPUT_DIR, "feature_norm.json"), "w") as f:
        json.dump(feat_norm, f, indent=2)

    train_feats = apply_feature_norm(train_feats_raw, feat_norm)
    val_feats = apply_feature_norm(val_df[feat_cols].astype(float), feat_norm)

    # labels
    def to_dir(y):
        return torch.tensor([LABELS.index(v) for v in y], dtype=torch.long)

    def to_ret(y):
        return torch.tensor(y.astype(float).values, dtype=torch.float)

    def to_abs(y):
        return torch.tensor(np.abs(y.astype(float).values), dtype=torch.float)

    y_dir_train = {
        30: to_dir(train_df["label_30m"]),
        60: to_dir(train_df["label_60m"]),
        120: to_dir(train_df["label_120m"]),
    }
    y_dir_val = {
        30: to_dir(val_df["label_30m"]),
        60: to_dir(val_df["label_60m"]),
        120: to_dir(val_df["label_120m"]),
    }
    y_ret_train = {
        30: to_ret(train_df["ret_30m"]),
        60: to_ret(train_df["ret_60m"]),
        120: to_ret(train_df["ret_120m"]),
    }
    y_ret_val = {
        30: to_ret(val_df["ret_30m"]),
        60: to_ret(val_df["ret_60m"]),
        120: to_ret(val_df["ret_120m"]),
    }

    y_abs_train = {
        30: to_abs(train_df["mag_30m"]),
        60: to_abs(train_df["mag_60m"]),
        120: to_abs(train_df["mag_120m"]),
    }
    y_abs_val = {
        30: to_abs(val_df["mag_30m"]),
        60: to_abs(val_df["mag_60m"]),
        120: to_abs(val_df["mag_120m"]),
    }

    # encodage texte
    enc_train = enc_texts(train_df)
    enc_val   = enc_texts(val_df)

    # tensors features
    Xf_train = torch.tensor(train_feats.values, dtype=torch.float)
    Xf_val   = torch.tensor(val_feats.values, dtype=torch.float)

    # datasets / loaders
    train_ds = MultiModalDataset(enc_train, Xf_train, y_dir_train, y_ret_train, y_abs_train)
    val_ds   = MultiModalDataset(enc_val,   Xf_val,   y_dir_val,   y_ret_val,   y_abs_val)

    # sampler pour l'imbalance direction (60m comme référence)
    y60 = y_dir_train[60]
    cnt = torch.bincount(y60, minlength=NUM_LABELS).double().clamp_min(1)
    inv = 1.0 / cnt
    samp_w = inv[y60]
    sampler = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ----- Model & losses -----
    feat_dim = Xf_train.shape[1]
    model = MultiModalHead(MODEL_NAME, feat_dim=feat_dim, hidden_drop=0.2).to(DEVICE)

    w30 = compute_class_weights(y_dir_train[30])
    w60 = compute_class_weights(y_dir_train[60])
    w120 = compute_class_weights(y_dir_train[120])

    if FOCAL_GAMMA and FOCAL_GAMMA > 0:
        dir30_loss = FocalLoss(weight=w30, gamma=FOCAL_GAMMA)
        dir60_loss = FocalLoss(weight=w60, gamma=FOCAL_GAMMA)
        dir120_loss = FocalLoss(weight=w120, gamma=FOCAL_GAMMA)
        print(f"[i] FocalLoss (gamma={FOCAL_GAMMA})")
    else:
        dir30_loss = nn.CrossEntropyLoss(weight=w30)
        dir60_loss = nn.CrossEntropyLoss(weight=w60)
        dir120_loss = nn.CrossEntropyLoss(weight=w120)

    ret_loss = nn.HuberLoss(delta=HUBER_DELTA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=max(1, int(total_steps * WARMUP_FRAC)),
        num_training_steps=total_steps)

    scaler = make_scaler(USE_AMP and DEVICE=="cuda")
    stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    best_key = -1.0

    # ----- Training loop -----
    for ep in range(1, EPOCHS+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {ep}", leave=True)
        tr_loss = 0.0

        for ids, mask, feats, y30, y60, y120, r30, r60, r120, a30, a60, a120 in loop:
            ids = ids.to(DEVICE); mask = mask.to(DEVICE); feats = feats.to(DEVICE)
            y30 = y30.to(DEVICE); y60 = y60.to(DEVICE); y120 = y120.to(DEVICE)
            r30 = r30.to(DEVICE); r60 = r60.to(DEVICE); r120 = r120.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(USE_AMP and DEVICE=="cuda"):
                (o30,o60,o120), (g30,g60,g120) = model(ids, mask, feats)
                loss_dir = dir30_loss(o30,y30) + dir60_loss(o60,y60) + dir120_loss(o120,y120)
                loss_ret = ret_loss(g30,r30) + ret_loss(g60,r60) + ret_loss(g120,r120)
                loss = LAMBDA_DIR*loss_dir + LAMBDA_RET*loss_ret

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tr_loss += float(loss.item())
            # Affichage séparé des pertes (ajout sans rien retirer)
            loop.set_postfix(loss=float(loss.item()),
                             loss_dir=float(loss_dir.item()),
                             loss_ret=float(loss_ret.item()))

        tr_loss /= max(1, len(train_loader))

        # Eval
        res = evaluate(model, val_loader, thresholds)
        acc_mean = (res[30]["dir_acc"] + res[60]["dir_acc"] + res[120]["dir_acc"]) / 3.0
        # clé de perf : direction en priorité, mais on log tout
        key = acc_mean

        msg = (f"[epoch {ep}] train_loss={tr_loss:.4f} | "
               f"dir_acc:30={res[30]['dir_acc']:.3f} 60={res[60]['dir_acc']:.3f} 120={res[120]['dir_acc']:.3f} | "
               f"dir_f1:30={res[30]['dir_f1']:.3f} 60={res[60]['dir_f1']:.3f} 120={res[120]['dir_f1']:.3f} | "
               f"ret_mae:30={res[30]['ret_mae']:.4f} 60={res[60]['ret_mae']:.4f} 120={res[120]['ret_mae']:.4f} | "
               f"mag_cls_acc:30={res[30]['mag_acc_cls']:.3f} 60={res[60]['mag_acc_cls']:.3f} 120={res[120]['mag_acc_cls']:.3f} | "
               f"ret_spr:30={res[30]['ret_spearman']:.3f} 60={res[60]['ret_spearman']:.3f} 120={res[120]['ret_spearman']:.3f}")
        print(msg)
        logf.write(msg + "\n"); logf.flush()
        save_jsonl(metrics_path, {"epoch": ep, "metrics": res, "train_loss": tr_loss})

        improved = stopper.step(key)
        if improved:
            best_key = key
            # save model & tokenizer
            ensure_dir(OUTPUT_DIR)
            model.backbone.save_pretrained(OUTPUT_DIR, safe_serialization=True)
            # on sauvegarde la totalité du modèle custom
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "multimodal_heads.pt"))
            tok.save_pretrained(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR,"best.json"),"w") as f:
                json.dump({"epoch": ep, "val_dir_mean_acc": best_key}, f)
            print(f"✅ Saved best (mean dir acc={best_key:.3f})")

        # Rapports direction (classification_report + CM)
        for h in [30,60,120]:
            yt = np.array(res[h]["dir_y_true"], int)
            yp = np.array(res[h]["dir_y_pred"], int)
            rep = classification_report(yt, yp, target_names=LABELS, digits=3)
            cm = confusion_matrix(yt, yp, labels=[0,1,2])
            with open(os.path.join(OUTPUT_DIR, f"classif_report_dir_{h}m.txt"), "w") as f:
                f.write(rep)
            np.savetxt(os.path.join(OUTPUT_DIR, f"cm_dir_{h}m.csv"), cm, fmt="%d", delimiter=",")

        # Rapports magnitude (classification via seuils train)
        for h in [30,60,120]:
            tm = np.array(res[h]["abs_y_true"], float)
            pm = np.array(res[h]["abs_y_pred"], float)
            q1 = thresholds[str(h)]["q1"]; q2 = thresholds[str(h)]["q2"]
            tm_cls = np.array([mag_to_class(x,q1,q2) for x in tm])
            pm_cls = np.array([mag_to_class(x,q1,q2) for x in pm])
            mask = tm_cls >= 0
            if mask.sum() > 0:
                rep = classification_report(tm_cls[mask], pm_cls[mask],
                                            target_names=["small","medium","large"], digits=3)
                cm = confusion_matrix(tm_cls[mask], pm_cls[mask], labels=[0,1,2])
                with open(os.path.join(OUTPUT_DIR, f"classif_report_mag_{h}m.txt"), "w") as f:
                    f.write(rep)
                np.savetxt(os.path.join(OUTPUT_DIR, f"cm_mag_{h}m.csv"), cm, fmt="%d", delimiter=",")

        if stopper.stop:
            print(f"[early stopping] patience={EARLY_STOP_PATIENCE} atteinte — stop.")
            break

    logf.close()
    print("[done] best mean dir acc =", round(best_key,3))

    # Sauvegarde automatique dans le log global
    from log_utils import append_run_log, default_params, print_training_log
    params = default_params(
        version="v7.1+",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        lambda_dir=LAMBDA_DIR,
        lambda_mag=LAMBDA_RET,
        gamma_focal=FOCAL_GAMMA,
        best_mean_dir_acc=best_key,
        best_dir_f1=(res[30]["dir_f1"] + res[60]["dir_f1"] + res[120]["dir_f1"]) / 3,
        val_loss=tr_loss,
        notes="direction prioritaire + régression retour signé"
    )
    append_run_log(params)
    print_training_log()

if __name__ == "__main__":
    main()