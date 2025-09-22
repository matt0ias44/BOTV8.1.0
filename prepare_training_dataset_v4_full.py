import pandas as pd
import numpy as np

# ======================
# Config
# ======================
NEWS_CSV = "btc_news_dataset.csv"         # news filtrées BTC (avec timestamp ok)
BTC_CSV  = "btcusdt_5m_full.csv"          # prix BTC 5m sur 3 ans
OUT_CSV  = "training_dataset_v4_full.csv"

ATR_WINDOW     = 6       # 6 * 5m = 30m
THRESH_COEFF   = 0.60    # seuil ATR pour bull/bear
MIN_MOVE_COEFF = 0.10    # filtrage mouvements trop petits

# ======================
# Load data
# ======================
print("[i] Chargement news...")
news = pd.read_csv(NEWS_CSV, low_memory=False)
news["timestamp"] = pd.to_datetime(news["timestamp"], errors="coerce", utc=True)
news = news.dropna(subset=["timestamp","title"]).sort_values("timestamp")
print(f"[i] News min/max dates: {news['timestamp'].min()} → {news['timestamp'].max()}")
print(f"[i] Total news chargées: {len(news)}")

print("[i] Chargement BTC 5m...")
btc = pd.read_csv(BTC_CSV)
btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce", utc=True)
btc = btc.dropna(subset=["timestamp","close"]).sort_values("timestamp").drop_duplicates("timestamp")
print(f"[i] BTC min/max dates: {btc['timestamp'].min()} → {btc['timestamp'].max()}")

# ======================
# ATR normalisé
# ======================
high, low, close = btc["high"].astype(float), btc["low"].astype(float), btc["close"].astype(float)
prev_close = close.shift(1)
tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
btc["atr_norm"] = (atr/close).clip(lower=1e-9)

# ======================
# Returns à horizons multiples
# ======================
btc = btc.set_index("timestamp")
for m in [30, 60, 120]:
    shift_n = m // 5  # nb bougies 5m
    btc[f"close_fut_{m}m"] = btc["close"].shift(-shift_n)
btc = btc.reset_index()

# Merge news avec prix & ATR
news = pd.merge_asof(news.sort_values("timestamp"),
                     btc[["timestamp","close","atr_norm"]].sort_values("timestamp"),
                     on="timestamp", direction="backward")

# Ajouter les retours futurs
for m in [30, 60, 120]:
    news = pd.merge_asof(news.sort_values("timestamp"),
                         btc[["timestamp",f"close_fut_{m}m"]].sort_values("timestamp"),
                         on="timestamp", direction="forward")
    news[f"return_{m}m"] = (news[f"close_fut_{m}m"] - news["close"]) / news["close"]

# ======================
# Labeling
# ======================
def label_adaptive(ret, atrn, c=THRESH_COEFF):
    if pd.isna(ret) or pd.isna(atrn): return "unknown"
    if ret >  c*atrn:  return "bullish"
    if ret < -c*atrn:  return "bearish"
    return "neutral"

news["label_30m_atr"] = [label_adaptive(r,a) for r,a in zip(news["return_30m"], news["atr_norm"])]

# ======================
# Filtrage mouvements trop plats
# ======================
mask = news["return_30m"].abs() >= (MIN_MOVE_COEFF * news["atr_norm"])
print(f"[i] Rows avant filtre : {len(news)}")
print(f"[i] Rows après filtre : {int(mask.sum())}")

news = news[mask].copy()

# ======================
# Export
# ======================
cols = [c for c in [
    "timestamp","title","sourceDomain","url",
    "return_30m","return_60m","return_120m",
    "atr_norm","label_30m_atr"
] if c in news.columns]

news[cols].to_csv(OUT_CSV, index=False)
print(f"✅ Export terminé : {OUT_CSV} ({len(news)} lignes gardées)")
