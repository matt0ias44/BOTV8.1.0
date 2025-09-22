#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge inference v2.
- Watches live_raw.csv produced by rss_to_csv.js
- Builds live market features via LiveFeatureBuilder
- Runs the multimodal model (direction + magnitude)
- Appends consolidated rows to live_predictions.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from fetch_article_bodies import fetch_article_body
from live.feature_builder import FeatureSource, LiveFeatureBuilder
from predictions_schema import EXPORTED_FEATURE_KEYS, PREDICTIONS_HEADER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models/bert_v7_1_plus"))
INPUT_CSV = Path("live_raw.csv")
OUTPUT_CSV = Path("live_predictions.csv")
PROCESSED_FILE = Path("live_processed_ids.json")
ARTICLE_CACHE_FILE = Path("live_article_cache.json")

LABELS = ["bearish", "neutral", "bullish"]
TARGET_HORIZON = "60"  # use 60m head as primary signal

EXPECTED_RAW_COLUMNS = [
    "datetime_paris",
    "datetime_utc",
    "title",
    "url",
    "summary",
    "source",
    "news_id",
]


MAX_ARTICLE_CACHE = 400


def load_article_cache() -> Dict[str, Dict[str, str]]:
    if not ARTICLE_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(ARTICLE_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"[WARN] unable to load article cache: {exc}")
    return {}


def save_article_cache(cache: Dict[str, Dict[str, str]]) -> None:
    items = sorted(
        cache.items(),
        key=lambda kv: kv[1].get("fetched_at", ""),
        reverse=True,
    )[:MAX_ARTICLE_CACHE]
    trimmed = {k: v for k, v in items}
    tmp = ARTICLE_CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ARTICLE_CACHE_FILE)


def fetch_article_with_cache(url: str, cache: Dict[str, Dict[str, str]]) -> tuple[str, str]:
    if not url:
        return "", "no_url"
    entry = cache.get(url)
    if entry:
        body = entry.get("body", "")
        status = entry.get("status", "cached" if body else "empty")
        return body, status
    body = fetch_article_body(url)
    status = "fetched" if body else "empty"
    cache[url] = {
        "body": body,
        "status": status,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    try:
        save_article_cache(cache)
    except Exception as exc:
        print(f"[WARN] unable to persist article cache: {exc}")
    return body, status


def build_text_payload(title: str, summary: str, body: str) -> tuple[str, str, str, str]:
    title_clean = (title or "").strip()
    summary_clean = (summary or "").strip()
    body_clean = (body or "").strip()

    titles_joined = title_clean
    body_concat_parts = [part for part in [summary_clean, body_clean] if part]
    body_concat = "\n\n".join(body_concat_parts) if body_concat_parts else title_clean
    text_parts = [part for part in [titles_joined, body_concat] if part]
    text_for_model = "\n\n".join(text_parts)
    if body_clean:
        source = "article"
    elif summary_clean:
        source = "summary"
    else:
        source = "title"
    return text_for_model or title_clean, titles_joined, body_concat, source


def _flatten_for_csv(value: str) -> str:
    if not value:
        return ""
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    return " ".join(normalized.split())


def _clean_field(value: str | float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _utc_to_paris(dt_utc: str) -> str:
    if not dt_utc:
        return ""
    ts = pd.to_datetime(dt_utc, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.tz_convert("Europe/Paris").isoformat()


def _ensure_news_id(dt_utc: str, title: str, existing: str) -> str:
    if existing:
        return existing
    base_dt = dt_utc.strip()
    base_title = title.strip()
    if base_dt and base_title:
        return f"{base_dt}|{base_title}"
    if base_title:
        return base_title
    return base_dt


def load_live_raw() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        return pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)

    rows: List[Dict[str, str]] = []
    with INPUT_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for idx, raw in enumerate(reader, start=2):
            if not raw:
                continue
            try:
                if len(raw) >= len(EXPECTED_RAW_COLUMNS):
                    mapped = {
                        col: _clean_field(value)
                        for col, value in zip(EXPECTED_RAW_COLUMNS, raw)
                    }
                elif len(raw) == 5:
                    dt_utc, title, url, summary, source = (_clean_field(v) for v in raw)
                    mapped = {
                        "datetime_utc": dt_utc,
                        "datetime_paris": _utc_to_paris(dt_utc),
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "source": source,
                        "news_id": "",
                    }
                else:
                    print(
                        f"[WARN] skipping malformed row {idx}: "
                        f"expected 5 or 7 columns, got {len(raw)}"
                    )
                    continue

                mapped["news_id"] = _ensure_news_id(
                    mapped.get("datetime_utc", ""),
                    mapped.get("title", ""),
                    _clean_field(mapped.get("news_id")),
                )
                if not mapped.get("datetime_paris"):
                    mapped["datetime_paris"] = _utc_to_paris(mapped.get("datetime_utc", ""))
                rows.append(mapped)
            except Exception as exc:
                print(f"[WARN] unable to parse row {idx}: {exc}")

    if not rows:
        return pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)

    df = pd.DataFrame(rows, columns=EXPECTED_RAW_COLUMNS)
    return df


class MultiModalHead(nn.Module):
    def __init__(self, model_name: str, feat_dim: int, hidden_drop: float = 0.2):
        super().__init__()
        load_kwargs = {"use_safetensors": True, "low_cpu_mem_usage": True}

        def _load_with_kwargs(kwargs: Dict) -> AutoModel:
            try:
                return AutoModel.from_pretrained(model_name, **kwargs)
            except OSError as err:
                if "paging file" in str(err).lower() or "1455" in str(err):
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("use_safetensors", None)
                    return AutoModel.from_pretrained(model_name, **retry_kwargs)
                raise

        dtype_kwargs: Dict[str, object] = {}
        if DEVICE == "cuda":
            dtype_kwargs["dtype"] = torch.float16

        try:
            self.backbone = _load_with_kwargs({**load_kwargs, **dtype_kwargs})
        except TypeError as err:
            if dtype_kwargs and ("dtype" in str(err) or "torch_dtype" in str(err)):
                legacy_kwargs = dict(load_kwargs)
                legacy_kwargs["torch_dtype"] = dtype_kwargs["dtype"]
                self.backbone = _load_with_kwargs(legacy_kwargs)
            else:
                raise
        hidden = self.backbone.config.hidden_size
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(hidden_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Dropout(hidden_drop),
        )
        self.dir30 = nn.Linear(hidden, 3)
        self.dir60 = nn.Linear(hidden, 3)
        self.dir120 = nn.Linear(hidden, 3)
        self.mag30 = nn.Linear(hidden, 1)
        self.mag60 = nn.Linear(hidden, 1)
        self.mag120 = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        fuse_dtype = next(self.fuse.parameters()).dtype
        if cls.dtype != fuse_dtype:
            cls = cls.to(fuse_dtype)
        mlp_dtype = next(self.feat_mlp.parameters()).dtype
        feats_for_mlp = feats.to(mlp_dtype) if feats.dtype != mlp_dtype else feats
        f = self.feat_mlp(feats_for_mlp)
        if f.dtype != fuse_dtype:
            f = f.to(fuse_dtype)
        z = self.fuse(torch.cat([cls, f], dim=-1))
        o_dir30 = self.dir30(z)
        o_dir60 = self.dir60(z)
        o_dir120 = self.dir120(z)
        o_mag30 = self.mag30(z).squeeze(-1)
        o_mag60 = self.mag60(z).squeeze(-1)
        o_mag120 = self.mag120(z).squeeze(-1)
        return (o_dir30, o_dir60, o_dir120), (o_mag30, o_mag60, o_mag120)


@dataclass
class InferenceAssets:
    model: MultiModalHead
    tokenizer: AutoTokenizer
    feat_norm: Dict[str, Dict[str, float]]
    feat_cols: List[str]
    thresholds: Optional[Dict[str, Dict[str, float]]]
    best_epoch: Optional[int]
    best_val_dir_mean_acc: Optional[float]
    best_metrics: Optional[Dict[str, Dict[str, float]]]


def load_model_assets(model_dir: Path) -> InferenceAssets:
    with open(model_dir / "feature_norm.json", "r", encoding="utf-8") as f:
        feat_norm = json.load(f)
    feat_cols = list(feat_norm.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = MultiModalHead(model_name=str(model_dir), feat_dim=len(feat_cols)).to(DEVICE)
    state_path = model_dir / "multimodal_heads.pt"
    if not state_path.exists():

        available = sorted(p.name for p in model_dir.glob("*.pt"))
        hint = (
            " Available checkpoints: " + ", ".join(available)
            if available
            else " No .pt files were found in the directory."
        )

        raise FileNotFoundError(
            "Model checkpoint not found. Expected to load "
            f"'{state_path}'. Either export the model weights by running "
            "`train_model_v7_1_multi.py` (which writes multimodal_heads.pt) "
            "or set the MODEL_DIR environment variable to point to a "
            "directory that contains the exported checkpoint."
        )




    try:
        state_dict = torch.load(
            state_path, map_location=DEVICE, weights_only=True  # type: ignore[arg-type]
        )
    except TypeError:
        # weights_only was introduced in newer torch releases. Fall back to the
        # legacy call signature when running on older versions so the code keeps
        # working without the new argument being available.
        state_dict = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    thresh_path = model_dir / "thresholds_mag.json"
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    best_epoch: Optional[int] = None
    best_val_dir_mean_acc: Optional[float] = None
    best_metrics: Optional[Dict[str, Dict[str, float]]] = None

    best_path = model_dir / "best.json"
    if best_path.exists():
        try:
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))
            best_epoch_val = best_payload.get("epoch")
            if isinstance(best_epoch_val, int):
                best_epoch = best_epoch_val
            best_metric_val = best_payload.get("val_dir_mean_acc")
            if isinstance(best_metric_val, (int, float)):
                best_val_dir_mean_acc = float(best_metric_val)
        except Exception as exc:
            print(f"[WARN] unable to read best.json: {exc}")

    metrics_summary: Optional[Dict[str, Dict[str, float]]] = None
    metrics_path = model_dir / "metrics_log.jsonl"
    if best_epoch is not None and metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if payload.get("epoch") == best_epoch:
                        metrics_raw = payload.get("metrics") or {}
                        metrics_summary = {}
                        for horizon_key, metrics_values in metrics_raw.items():
                            # json may encode horizon as string; ensure str keys
                            horizon = str(horizon_key)
                            if not isinstance(metrics_values, dict):
                                continue
                            filtered: Dict[str, float] = {}
                            for metric_name in (
                                "dir_acc",
                                "dir_f1",
                                "ret_mae",
                                "mag_acc_cls",
                                "mag_f1_cls",
                                "ret_spearman",
                            ):
                                metric_val = metrics_values.get(metric_name)
                                if isinstance(metric_val, (int, float)):
                                    filtered[metric_name] = float(metric_val)
                            if filtered:
                                metrics_summary[horizon] = filtered
                        break
        except Exception as exc:
            print(f"[WARN] unable to parse metrics_log.jsonl: {exc}")

    if metrics_summary:
        best_metrics = metrics_summary

    if best_epoch is not None or best_val_dir_mean_acc is not None or best_metrics:
        summary_parts: List[str] = []
        if best_epoch is not None:
            summary_parts.append(f"epoch={best_epoch}")
        if best_val_dir_mean_acc is not None:
            summary_parts.append(f"val_dir_mean_acc={best_val_dir_mean_acc:.3f}")
        if best_metrics:
            for horizon in ("30", "60", "120"):
                horizon_metrics = best_metrics.get(horizon)
                if not horizon_metrics:
                    continue
                horizon_parts = [
                    f"{horizon}m dir_acc={horizon_metrics.get('dir_acc', float('nan')):.3f}",
                    f"dir_f1={horizon_metrics.get('dir_f1', float('nan')):.3f}",
                    f"ret_mae={horizon_metrics.get('ret_mae', float('nan')):.4f}",
                ]
                if "mag_acc_cls" in horizon_metrics:
                    horizon_parts.append(
                        f"mag_acc={horizon_metrics.get('mag_acc_cls', float('nan')):.3f}"
                    )
                summary_parts.append(" | ".join(horizon_parts))
        print("[model] Loaded best checkpoint: " + " || ".join(summary_parts))

    return InferenceAssets(
        model=model,
        tokenizer=tokenizer,
        feat_norm=feat_norm,
        feat_cols=feat_cols,
        thresholds=thresholds,
        best_epoch=best_epoch,
        best_val_dir_mean_acc=best_val_dir_mean_acc,
        best_metrics=best_metrics,
    )


def normalize_features(feat_norm: Dict[str, Dict[str, float]], feats: Dict[str, float], feat_cols: List[str]):
    vec = []
    for col in feat_cols:
        stats = feat_norm[col]
        val = feats.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = stats.get("mean", 0.0)
        vec.append((val - stats["mean"]) / (stats["std"] or 1e-12))
    return torch.tensor([vec], dtype=torch.float32, device=DEVICE)


def fallback_features(feat_norm: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {col: stats.get("mean", 0.0) for col, stats in feat_norm.items()}


def magnitude_bucket(thresholds: Dict[str, Dict[str, float]] | None, value: float) -> str:
    if thresholds is None:
        return "unknown"
    bucket = thresholds.get(TARGET_HORIZON)
    if not bucket:
        return "unknown"
    q1 = bucket.get("q1", 0.0)
    q2 = bucket.get("q2", 0.0)
    if np.isnan(value):
        return "unknown"
    abs_val = abs(value)
    if abs_val < q1:
        return "small"
    if abs_val < q2:
        return "medium"
    return "large"


def ensure_output_header():
    expected_header = ",".join(PREDICTIONS_HEADER)

    if not OUTPUT_CSV.exists():
        OUTPUT_CSV.write_text(expected_header + "\n", encoding="utf-8")
        return

    try:
        with OUTPUT_CSV.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"[WARN] unable to read {OUTPUT_CSV}: {exc}")
        return

    current_line = lines[0].strip() if lines else ""
    current_header = [part.strip() for part in current_line.split(",") if part.strip()]

    if current_header == PREDICTIONS_HEADER:
        if not lines:
            OUTPUT_CSV.write_text(expected_header + "\n", encoding="utf-8")
        return

    tmp_path = OUTPUT_CSV.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(expected_header + "\n")
        if len(lines) > 1:
            fh.writelines(lines[1:])
    tmp_path.replace(OUTPUT_CSV)
    print(
        f"[WARN] rewrote {OUTPUT_CSV} header to expected schema "
        f"({len(current_header)} cols -> {len(PREDICTIONS_HEADER)})."
    )


def load_processed_ids() -> List[str]:
    if not PROCESSED_FILE.exists():
        return []
    try:
        ids = json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
        if isinstance(ids, list):
            return ids
    except Exception as exc:
        print(f"[WARN] could not read {PROCESSED_FILE}: {exc}")
    return []


def persist_processed_ids(ids: List[str]):
    keep = ids[-5000:]
    PROCESSED_FILE.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")


def tail_news(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 200:
        return df
    return df.tail(200)


def run_loop():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    assets = load_model_assets(MODEL_DIR)
    feature_builder = LiveFeatureBuilder(FeatureSource(), feat_stats=assets.feat_norm)
    processed_ids = load_processed_ids()
    processed_set = set(processed_ids)
    ensure_output_header()
    article_cache = load_article_cache()

    print("[INFO] Bridge inference v2 started. Watching", INPUT_CSV)

    while True:
        try:
            if not INPUT_CSV.exists():
                time.sleep(2)
                continue

            df = load_live_raw()
            if df.empty:
                time.sleep(2)
                continue

            df = tail_news(df)
            if "news_id" not in df.columns:
                time.sleep(2)
                continue

            for _, row in df.iterrows():
                news_id = str(row.get("news_id", "").strip())
                if not news_id or news_id in processed_set:
                    continue

                title = str(row.get("title", ""))
                summary = str(row.get("summary", ""))
                url = str(row.get("url", ""))
                source = str(row.get("source", ""))

                article_body, article_status = fetch_article_with_cache(url, article_cache)
                text_for_model, titles_joined, body_concat, text_source = build_text_payload(
                    title,
                    summary,
                    article_body,
                )
                article_chars = len(article_body.strip()) if article_body else 0
                article_found = bool(article_body and article_chars >= 200)
                if article_found:
                    article_status = f"{article_status}_full"
                elif article_body:
                    article_status = f"{article_status}_short"
                elif summary:
                    article_status = f"{article_status}_summary"
                else:
                    article_status = f"{article_status}_title"
                text_chars = len(text_for_model)

                dt_paris = pd.to_datetime(row.get("datetime_paris"), utc=True, errors="coerce")
                dt_utc = pd.to_datetime(row.get("datetime_utc"), utc=True, errors="coerce")
                if pd.isna(dt_utc):
                    dt_utc = dt_paris.tz_convert("UTC") if dt_paris is not None else pd.Timestamp.utcnow().tz_localize("UTC")
                event_time = dt_utc.to_pydatetime()

                features_status = "live"
                sentiment_val = 0.0
                for sent_key in ("sentiment_score", "sentiment"):
                    raw_sent = row.get(sent_key)
                    if raw_sent is None or raw_sent == "":
                        continue
                    try:
                        sentiment_val = float(raw_sent)
                        break
                    except (TypeError, ValueError):
                        continue
                try:
                    feats = feature_builder.build(event_time, sentiment=sentiment_val)
                except Exception as exc:
                    print(f"[WARN] feature builder failed for {news_id}: {exc}; fallback to means")
                    feats = fallback_features(assets.feat_norm)
                    features_status = "fallback_means"

                Xf = normalize_features(assets.feat_norm, feats, assets.feat_cols)

                enc = assets.tokenizer(
                    text_for_model,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(DEVICE)
                mask = enc["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    (o30, o60, o120), (g30, g60, g120) = assets.model(ids, mask, Xf)

                logits = o60
                probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
                pred_idx = int(probs.argmax())
                label = LABELS[pred_idx]
                confidence = float(probs[pred_idx])

                ret_30 = float(g30.cpu().numpy().reshape(-1)[0])
                ret_60 = float(g60.cpu().numpy().reshape(-1)[0])
                ret_120 = float(g120.cpu().numpy().reshape(-1)[0])
                mag_val = abs(ret_60)
                bucket = magnitude_bucket(assets.thresholds, ret_60)

                feature_exports = {key: feats.get(key) for key in EXPORTED_FEATURE_KEYS}


                processed_ids.append(news_id)
                processed_set.add(news_id)
                if len(processed_ids) > 6000:
                    processed_ids = processed_ids[-5000:]
                    processed_set = set(processed_ids)
                persist_processed_ids(processed_ids)

                out_row = {
                    "news_id": news_id,
                    "datetime_paris": row.get("datetime_paris"),
                    "datetime_utc": dt_utc.isoformat(),
                    "prediction": label,
                    "prob_bear": round(float(probs[0]), 6),
                    "prob_neut": round(float(probs[1]), 6),
                    "prob_bull": round(float(probs[2]), 6),
                    "confidence": round(confidence, 6),
                    "ret_pred": round(ret_60, 6),
                    "ret_30m_pred": round(ret_30, 6),
                    "ret_120m_pred": round(ret_120, 6),
                    "mag_pred": round(mag_val, 6),
                    "mag_30m_pred": round(abs(ret_30), 6),
                    "mag_120m_pred": round(abs(ret_120), 6),
                    "mag_bucket": bucket,
                    "features_status": features_status,
                    "title": _flatten_for_csv(title),
                    "summary": _flatten_for_csv(summary),
                    "url": url,
                    "source": source,
                    "titles_joined": _flatten_for_csv(titles_joined),
                    "body_concat": _flatten_for_csv(body_concat),
                    "text_source": text_source,
                    "text_chars": int(text_chars),
                    "article_status": article_status,
                    "article_found": bool(article_found),
                    "article_chars": int(article_chars),
                    "processed_at": pd.Timestamp.utcnow().isoformat(),
                }
                out_row.update({k: (None if v is None else float(v)) for k, v in feature_exports.items()})

                df_out = pd.DataFrame([out_row])
                df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                print(
                    f"[PRED] {news_id} | {label} conf={confidence:.2f} "
                    f"bull={probs[2]:.2f} bear={probs[0]:.2f} ret={ret_60:.4f} mag={mag_val:.4f}"
                )
        except Exception as exc:
            print("[ERROR]", exc)
            time.sleep(5)

        time.sleep(2)


if __name__ == "__main__":
    run_loop()
