#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_event_labels_v2
========================

Génère des cibles multi-horizons à partir d'un historique OHLCV 1 minute.

Pour chaque événement, on calcule :

* le label directionnel (bearish / neutral / bullish) basé sur un seuil dynamique
  proportionnel à l'ATR courant,
* le rendement signé ``ret_{h}m`` (pct_change) pour chaque horizon,
* la magnitude absolue ``mag_{h}m`` (``abs(ret_{h}m)``).

Ces informations permettent ensuite d'entraîner un modèle multi-tâches qui
prévoit simultanément la direction et l'amplitude du mouvement.
"""

import argparse
from datetime import timedelta
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

# =====================
# Helpers
# =====================
def as_utc_series(s: pd.Series):
    return pd.to_datetime(s, utc=True, errors="coerce")

ATR_COL = "atr_30m_pct"
RET_THRESH_BASE = 0.0005  # 0.05%
RET_THRESH_ATR_MULT = 0.35  # 35% de l'ATR pour déclencher bullish/bearish


def compute_direction(ret: float, atr_pct: Optional[float]) -> str:
    """Convertit un rendement en label directionnel."""

    thresh = RET_THRESH_BASE
    if atr_pct is not None and not np.isnan(atr_pct):
        thresh = max(thresh, float(atr_pct) * RET_THRESH_ATR_MULT)

    if ret > thresh:
        return "bullish"
    if ret < -thresh:
        return "bearish"
    return "neutral"


def compute_labels(ev_time: pd.Timestamp, prices: pd.DataFrame, horizons: Iterable[int]) -> Dict[str, Union[float, str]]:
    """Calcule labels directionnels, rendements et magnitudes pour un événement."""

    out: Dict[str, Union[float, str]] = {}
    if ev_time.tzinfo is None:
        ev_time = ev_time.tz_localize("UTC")

    try:
        base_row = prices.loc[:ev_time].iloc[-1]
    except IndexError:
        for h in horizons:
            out[f"label_{h}m"] = np.nan
            out[f"ret_{h}m"] = np.nan
            out[f"mag_{h}m"] = np.nan
        return out

    p0 = float(base_row["close"])
    atr_pct = float(base_row.get(ATR_COL, np.nan))

    for h in horizons:
        dt_future = ev_time + timedelta(minutes=int(h))
        try:
            p1 = float(prices.loc[:dt_future].iloc[-1]["close"])
        except IndexError:
            out[f"label_{h}m"] = np.nan
            out[f"ret_{h}m"] = np.nan
            out[f"mag_{h}m"] = np.nan
            continue

        ret = (p1 - p0) / max(p0, 1e-9)
        out[f"label_{h}m"] = compute_direction(ret, atr_pct)
        out[f"ret_{h}m"] = ret
        out[f"mag_{h}m"] = abs(ret)

    return out

# =====================
# Main
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="CSV des events (news GDELT ou interne)")
    ap.add_argument("--ohlcv", required=True, help="CSV OHLCV BTC 1m")
    ap.add_argument("--out", required=True, help="CSV de sortie avec labels")
    args = ap.parse_args()

    print("[i] Chargement des events...")
    ev = pd.read_csv(args.events)

    # Normalisation des colonnes
    if "titles_joined" not in ev.columns and "title" in ev.columns:
        ev["titles_joined"] = ev["title"]
    if "body_concat" not in ev.columns and "body" in ev.columns:
        ev["body_concat"] = ev["body"]

    if "event_time" not in ev.columns:
        raise ValueError("Le CSV d'events doit contenir 'event_time'")
    ev["event_time"] = as_utc_series(ev["event_time"])
    ev = ev.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    print("[i] Chargement OHLCV...")
    prices = pd.read_csv(args.ohlcv)
    prices = prices.rename(columns={c: c.lower() for c in prices.columns})

    if "open_time" in prices.columns:
        prices["ts"] = pd.to_datetime(prices["open_time"], unit="ms", utc=True, errors="coerce")
    elif "ts" in prices.columns:
        prices["ts"] = pd.to_datetime(prices["ts"], utc=True, errors="coerce")
    elif "timestamp" in prices.columns:
        ts_raw = prices["timestamp"]
        if np.issubdtype(ts_raw.dtype, np.number):
            unit = "ms" if ts_raw.max() > 1e11 else "s"
            prices["ts"] = pd.to_datetime(ts_raw, unit=unit, utc=True, errors="coerce")
        else:
            prices["ts"] = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    else:
        raise ValueError("OHLCV doit contenir 'open_time', 'ts' ou 'timestamp'")

    if "close" not in prices.columns:
        raise ValueError("OHLCV doit contenir une colonne close")

    prices = (
        prices.dropna(subset=["ts"])
        .drop_duplicates(subset=["ts"])
        .set_index("ts")
        .sort_index()
    )

    # Calcul ATR & volatilité réalisée pour définir des seuils dynamiques
    high = prices.get("high", prices["close"])
    low = prices.get("low", prices["close"])
    close = prices["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_30 = tr.rolling(30).mean()
    atr_ratio = atr_30 / close
    prices[ATR_COL] = atr_ratio.ffill()

    # Remplissage forward pour éviter les trous dans l'index temporel
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq="1min", tz="UTC")
    prices = prices.reindex(full_idx).ffill()

    # Calcul des labels
    print("[i] Calcul des labels...")
    labels = []
    for _, row in ev.iterrows():
        labs = compute_labels(row["event_time"], prices, horizons=[30, 60, 120])
        labels.append(labs)
    lab_df = pd.DataFrame(labels)

    out = pd.concat([ev, lab_df], axis=1)
    out.to_csv(args.out, index=False)
    print(f"[ok] Sauvegardé -> {args.out}, {len(out)} lignes")

if __name__ == "__main__":
    main()