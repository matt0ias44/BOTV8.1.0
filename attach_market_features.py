#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ajoute des features de marché à t0 pour chaque événement (sans fuite).
Réutilisable comme fonction dans live_pipeline.
"""

import math

import numpy as np
import pandas as pd


def as_utc(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    if s.isna().any():
        raise ValueError("Erreur de parsing des timestamps.")
    return s


def rsi(series, period=14):
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def realized_vol(ret1m, window):
    return ret1m.rolling(window).std()


def annualize_vol(vol, window):
    if window <= 0:
        return vol
    scale = math.sqrt(1440 / window)
    return vol * scale


def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def attach_features_to_df(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features marché et contextuelles à chaque événement."""
    ev = ev.copy()
    ev["event_time"] = as_utc(ev["event_time"])
    px["timestamp"] = as_utc(px["timestamp"])
    px = px.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")

    # timeline complète en 1m
    full_idx = pd.date_range(px.index.min(), px.index.max(), freq="1min", tz="UTC")
    px = px.reindex(full_idx).ffill()

    # Retours
    px["ret1m"] = px["close"].pct_change()
    px["logret1m"] = np.log(px["close"] / px["close"].shift(1))

    # Volatilité réalisée multi-horizons (simple et annualisée)
    for w in [30, 60, 120, 240]:
        rv = realized_vol(px["logret1m"], w)
        px[f"realized_vol_{w}m"] = rv
        px[f"realized_vol_{w}m_annual"] = annualize_vol(rv, w)

    # RSI
    px["rsi_14"] = rsi(px["close"], period=14)

    # ATR + version normalisée par le prix
    tr = true_range(px)
    for w in [30, 60, 120]:
        atr = tr.rolling(w).mean()
        px[f"atr_{w}m"] = atr
        px[f"atr_{w}m_pct"] = atr / (px["close"].rolling(1).mean() + 1e-12)

    # Largeur de bandes de Bollinger (20 périodes)
    mid = px["close"].rolling(20).mean()
    std = px["close"].rolling(20).std()
    px["boll_width_20"] = (2 * std) / (mid + 1e-12)

    # Momentum prix
    for w in [5, 15, 30, 60]:
        px[f"momentum_{w}m"] = px["close"].pct_change(w)

    # Volume et liquidité
    px["vol_z_60m"] = (px["volume"] - px["volume"].rolling(60).mean()) / (px["volume"].rolling(60).std() + 1e-12)
    px["volume_rate_30m"] = px["volume"].rolling(5).sum() / (px["volume"].rolling(30).sum() + 1e-12)
    px["volume_trend_120m"] = px["volume"].ewm(span=120, adjust=False).mean()

    # Moyennes mobiles
    px["sma_10"] = px["close"].rolling(10).mean()
    px["sma_50"] = px["close"].rolling(50).mean()
    px["sma10_sma50_diff"] = (px["sma_10"] - px["sma_50"]) / (px["sma_50"] + 1e-12)

    # Signatures temporelles (sin/cos heure, jour semaine)
    idx = px.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    hour_float = idx.hour + idx.minute / 60.0
    px["intraday_sin"] = np.sin(2 * np.pi * hour_float / 24)
    px["intraday_cos"] = np.cos(2 * np.pi * hour_float / 24)
    px["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    px["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    px["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    feat_cols = [c for c in px.columns if c not in ["open", "high", "low", "close", "volume", "ret1m", "logret1m"]]

    def get_feats_at(t):
        try:
            return px.loc[:t].iloc[-1][feat_cols]
        except Exception:
            return pd.Series({c: np.nan for c in feat_cols})

    feats = pd.DataFrame([get_feats_at(t) for t in ev["event_time"].tolist()]).reset_index(drop=True)
    feats.columns = [f"feat_{c}" for c in feats.columns]

    # Features sentiment / flux d'event (agrégation locale)
    ev = ev.reset_index(drop=True)
    ev["event_time"] = as_utc(ev["event_time"])
    ev = ev.sort_values("event_time").reset_index(drop=True)

    if "sentiment_score" in ev.columns:
        sent = pd.to_numeric(ev["sentiment_score"], errors="coerce").fillna(0.0)
    elif "sentiment" in ev.columns:
        sent = pd.to_numeric(ev["sentiment"], errors="coerce").fillna(0.0)
    else:
        sent = pd.Series(np.zeros(len(ev)))

    sent_roll = sent.rolling(window=10, min_periods=1).mean().shift(1).fillna(0.0)
    sent_std = sent.rolling(window=10, min_periods=2).std().shift(1).fillna(0.0)

    ev_idx = ev.set_index("event_time")
    ones = pd.Series(1.0, index=ev_idx.index)
    flux_60 = ones.rolling("60min", closed="left").sum().fillna(0.0).values
    flux_180 = ones.rolling("180min", closed="left").sum().fillna(0.0).values

    context_feats = pd.DataFrame(
        {
            "feat_sent_roll_mean_10": sent_roll.values,
            "feat_sent_roll_std_10": sent_std.values,
            "feat_news_count_60m": flux_60,
            "feat_news_count_180m": flux_180,
        }
    )

    out = pd.concat([ev.reset_index(drop=True), feats, context_feats], axis=1)
    return out


if __name__ == "__main__":
    EVENTS_FILE = "events_with_magnitude.csv"
    PRICE_FILE = "btcusdt_1m_full.csv"
    OUT_FILE = "events_with_features.csv"

    print("[i] Chargement...")
    ev = pd.read_csv(EVENTS_FILE)
    px = pd.read_csv(PRICE_FILE)

    out = attach_features_to_df(ev, px)
    out.to_csv(OUT_FILE, index=False)
    print(f"[ok] écrit {OUT_FILE}")