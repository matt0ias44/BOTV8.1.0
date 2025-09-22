#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit dashboard for the BTC sentiment trading bot."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import warnings
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pytz import timezone as pytz_timezone
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore", message=".*Styler.applymap.*", category=FutureWarning)

TZ_PARIS = pytz_timezone("Europe/Paris")
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "models/bert_v7_1_plus"

PREDICTIONS_CSV = APP_ROOT / "live_predictions.csv"
STATE_FILE = APP_ROOT / "bot_state.json"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_PATH)))
LIVE_RAW_FILE = APP_ROOT / "live_raw.csv"

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_24H_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
BINANCE_PRICE_EUR_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCEUR"
BINANCE_24H_EUR_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCEUR"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
REFRESH_SEC = 5

DEFAULT_STATE = {
    "starting_equity": 10000.0,
    "equity": 10000.0,
    "position": None,
    "trades": [],
    "equity_curve": [],
    "last_pred_id": None,
    "last_signal": None,
}


def humanize_delta(delta: timedelta) -> str:
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "moins d'une minute"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    minutes %= 60
    if hours < 24:
        return f"{hours} h {minutes:02d}"
    days = hours // 24
    hours %= 24
    return f"{days} j {hours:02d} h"


def file_recent_status(path: Path, threshold_minutes: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "fichier manquant"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - mtime
    ok = age <= timedelta(minutes=threshold_minutes)
    detail = f"maj il y a {humanize_delta(age)}"
    return ok, detail


def check_rss_status() -> tuple[bool, str]:
    ok, detail = file_recent_status(LIVE_RAW_FILE, 30)
    return ok, detail


def check_model_status(preds: pd.DataFrame) -> tuple[bool, str]:
    if not MODEL_DIR.exists():
        return False, "répertoire modèle absent"
    if not PREDICTIONS_CSV.exists():
        return False, "live_predictions.csv manquant"
    if preds.empty:
        ok, detail = file_recent_status(PREDICTIONS_CSV, 30)
        return ok, detail + " (aucune ligne)"
    if "datetime_utc" in preds.columns and preds["datetime_utc"].notna().any():
        last_dt = preds["datetime_utc"].dropna().iloc[-1]
        last_dt = pd.to_datetime(last_dt, utc=True)
        age = datetime.now(tz=timezone.utc) - last_dt.to_pydatetime()
    else:
        mtime = datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(tz=timezone.utc) - mtime
    ok = age <= timedelta(minutes=30)
    detail = f"dernière prédiction il y a {humanize_delta(age)}"
    return ok, detail


def check_trader_status(state: Dict[str, Any]) -> tuple[bool, str]:
    if not STATE_FILE.exists():
        return False, "bot_state.json manquant"
    curve = state.get("equity_curve") or []
    if not curve:
        ok, detail = file_recent_status(STATE_FILE, 10)
        return ok, detail + " (courbe vide)"
    last_ts = curve[-1][0]
    dt = pd.to_datetime(last_ts, utc=True, errors="coerce")
    if pd.isna(dt):
        return False, "horodatage equity invalide"
    age = datetime.now(tz=timezone.utc) - dt.to_pydatetime()
    ok = age <= timedelta(minutes=30)
    detail = f"maj trader il y a {humanize_delta(age)}"
    return ok, detail


def render_status_badge(column, label: str, ok: bool, detail: str) -> None:
    color = "#16c784" if ok else "#ea3943"
    status = "OK" if ok else "OFF"
    column.markdown(
        f"""
        <div style="background:#111;border:1px solid #333;border-radius:10px;padding:10px 12px;margin-bottom:6px;">
          <div style="color:#bbb;font-size:12px;margin-bottom:4px;">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:700;">{status}</div>
          <div style="color:#777;font-size:11px;margin-top:3px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def now_paris_str() -> str:
    return datetime.now(TZ_PARIS).strftime("%d/%m/%Y %H:%M:%S")


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for key, value in DEFAULT_STATE.items():
                state.setdefault(key, value)
            return state
        except Exception as exc:
            st.warning(f"Unable to load bot_state.json: {exc}")
    return DEFAULT_STATE.copy()


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
    except Exception as exc:
        st.error(f"Unable to read {PREDICTIONS_CSV}: {exc}")
        return pd.DataFrame()

    if "datetime_paris" in df.columns:
        df["datetime_paris"] = pd.to_datetime(df["datetime_paris"], utc=True, errors="coerce").dt.tz_convert(TZ_PARIS)
    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    return df


LABEL_CHOICES = {"bearish", "neutral", "bullish"}

LABEL_TRANSLATIONS = {
    "bullish": "Haussier",
    "bearish": "Baissier",
    "neutral": "Neutre",
}



def _as_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    val = _as_float(value)
    if val is None:
        return None
    return int(round(val))



def _as_bool(value: Any) -> Optional[bool]:
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        if lowered in {"nan", "none", "null"}:
            return None
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):  # type: ignore[arg-type]
                return None
        except TypeError:
            pass
        return bool(value)
    return None


def _clean_text(value: Any) -> str:
    if value in (None, "", "None"):
        return ""
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"none", "nan", "null"}:
        return ""
    try:
        float(text)
    except (TypeError, ValueError):
        return text
    return ""


def _normalize_label(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    aliases = {
        "bull": "bullish",
        "buy": "bullish",
        "long": "bullish",
        "bear": "bearish",
        "sell": "bearish",
        "short": "bearish",
    }
    text = aliases.get(text, text)
    if text in LABEL_CHOICES:
        return text
    return ""


def _probabilities_from_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    aliases = {
        "bullish": ["prob_bull", "prob_pos", "prob_long"],
        "bearish": ["prob_bear", "prob_neg", "prob_short"],
        "neutral": ["prob_neut", "prob_neutral"],
    }
    result: Dict[str, float] = {}
    for label, keys in aliases.items():
        for key in keys:
            val = _as_float(payload.get(key))
            if val is None:
                continue
            try:
                if np.isnan(val):  # type: ignore[arg-type]
                    continue
            except TypeError:
                pass
            result[label] = val
            break
    return result


def _best_label_from_probs(payload: Dict[str, Any]) -> tuple[str, Optional[float]]:
    cleaned = _probabilities_from_payload(payload)
    if not cleaned:
        return "", None
    label, value = max(cleaned.items(), key=lambda item: item[1])
    return label, value


def resolve_last_signal(preds: pd.DataFrame, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    latest: Dict[str, Any] = {}
    row_dict: Dict[str, Any] = {}
    probs_from_row: Dict[str, float] = {}

    if preds is not None and not preds.empty:
        row_dict = preds.iloc[-1].to_dict()
        probs_from_row = _probabilities_from_payload(row_dict)

        pred_label = _normalize_label(row_dict.get('prediction'))
        if not pred_label and probs_from_row:
            pred_label = max(probs_from_row, key=probs_from_row.get)
        if pred_label:
            latest['prediction'] = pred_label

        conf_from_row = _as_float(row_dict.get('confidence')) if 'confidence' in row_dict else None
        if conf_from_row is not None:
            latest['confidence'] = conf_from_row
            latest['confidence_source'] = 'predictions'
        elif probs_from_row:
            best_label = max(probs_from_row, key=probs_from_row.get)
            latest.setdefault('prediction', best_label)
            latest['confidence'] = probs_from_row[best_label]
            latest['confidence_source'] = 'probabilities'

        if probs_from_row:
            latest['probabilities'] = probs_from_row

        ret_val = _as_float(row_dict.get('ret_pred'))
        if ret_val is not None:
            latest['ret_pred'] = ret_val
            latest['ret_source'] = 'predictions'

        mag_val = _as_float(row_dict.get('mag_pred'))
        if mag_val is not None:
            latest['mag_pred'] = mag_val
            latest['mag_source'] = 'predictions'

        article_status = _clean_text(row_dict.get('article_status'))
        if article_status:
            latest['article_status'] = article_status
        article_found = _as_bool(row_dict.get('article_found'))
        if article_found is not None:
            latest['article_found'] = article_found
            latest['article_found_source'] = 'predictions'
        article_chars = _as_int(row_dict.get('article_chars'))
        if article_chars is not None:
            latest['article_chars'] = article_chars
        text_source = _clean_text(row_dict.get('text_source'))
        if text_source:
            latest['text_source'] = text_source
        title_val = _clean_text(row_dict.get('title'))
        if title_val:
            latest['title'] = title_val
        url_val = _clean_text(row_dict.get('url'))
        if url_val:
            latest['url'] = url_val
        features_status = _clean_text(row_dict.get('features_status'))
        if features_status:
            latest['features_status'] = features_status
        news_id = _clean_text(row_dict.get('news_id'))
        if news_id:
            latest['news_id'] = news_id

        event_time = row_dict.get('datetime_paris') or row_dict.get('datetime_utc') or row_dict.get('processed_at')
        if isinstance(event_time, str) and event_time:
            try:
                event_time = pd.to_datetime(event_time, utc=True)
            except Exception:
                event_time = None
        if isinstance(event_time, pd.Timestamp):
            latest['event_time'] = event_time

    state_sig = state.get('last_signal') if isinstance(state, dict) else None
    if isinstance(state_sig, dict) and state_sig:
        state_pred = _normalize_label(state_sig.get('prediction'))
        if state_pred:
            latest['prediction'] = state_pred

        for key in ('confidence', 'ret_pred', 'mag_pred'):
            val = _as_float(state_sig.get(key))
            if val is not None and latest.get(key) is None:
                latest[key] = val
                latest[f"{key}_source"] = 'state'

        lev = _as_int(state_sig.get('planned_leverage'))
        if lev is not None and lev > 0 and latest.get('planned_leverage') is None:
            latest['planned_leverage'] = lev
            latest['planned_leverage_source'] = 'state'

        risk = _as_float(state_sig.get('risk_fraction'))
        if risk is not None and latest.get('risk_fraction') is None:
            latest['risk_fraction'] = risk
            latest['risk_fraction_source'] = 'state'

        for key in ('article_status', 'text_source', 'title', 'url', 'features_status', 'news_id'):
            val = _clean_text(state_sig.get(key))
            if val and not _clean_text(latest.get(key)):
                latest[key] = val

        art_found_state = _as_bool(state_sig.get('article_found'))
        if art_found_state is not None and latest.get('article_found') is None:
            latest['article_found'] = art_found_state
            latest['article_found_source'] = 'state'

        art_chars_state = _as_int(state_sig.get('article_chars'))
        if art_chars_state is not None and latest.get('article_chars') is None:
            latest['article_chars'] = art_chars_state

        time_val = state_sig.get('time') or state_sig.get('event_time')
        if time_val and 'event_time' not in latest:
            try:
                latest['event_time'] = pd.to_datetime(time_val, utc=True)
            except Exception:
                latest['event_time'] = time_val

    if not latest:
        return None

    normalized = _normalize_label(latest.get('prediction'))
    if not normalized:
        fallback_label, fallback_conf = _best_label_from_probs(row_dict) if row_dict else ('', None)
        if fallback_label:
            latest['prediction'] = fallback_label
            if fallback_conf is not None and latest.get('confidence') is None:
                latest['confidence'] = fallback_conf
                latest['confidence_source'] = latest.get('confidence_source', 'probabilities')
        else:
            latest['prediction'] = 'neutral'
    else:
        latest['prediction'] = normalized

    conf_val = _as_float(latest.get('confidence'))
    if conf_val is not None:
        latest['confidence'] = conf_val
    else:
        latest.pop('confidence', None)

    return latest


def fetch_btc_ticker() -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    try:
        pr = requests.get(BINANCE_PRICE_URL, timeout=4).json()
        s24 = requests.get(BINANCE_24H_URL, timeout=4).json()
        result.update(
            price=float(pr.get("price")),
            change_pct=float(s24.get("priceChangePercent")),
            high=float(s24.get("highPrice")),
            low=float(s24.get("lowPrice")),
            volume=float(s24.get("volume")),
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    try:
        pr_eur = requests.get(BINANCE_PRICE_EUR_URL, timeout=4).json()
        s24_eur = requests.get(BINANCE_24H_EUR_URL, timeout=4).json()
        result["price_eur"] = float(pr_eur.get("price"))
        result["change_pct_eur"] = float(s24_eur.get("priceChangePercent"))
        result["high_eur"] = float(s24_eur.get("highPrice"))
        result["low_eur"] = float(s24_eur.get("lowPrice"))
    except Exception:
        result.setdefault("price_eur", None)
    return result


def fetch_klines(interval: str = "1m", limit: int = 300) -> pd.DataFrame:
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    rows = [
        {
            "open_time": datetime.fromtimestamp(item[0] / 1000, tz=TZ_PARIS),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in data
    ]
    return pd.DataFrame(rows)


def kpi_block(label: str, value: str, help_text: Optional[str] = None, color: str = "white") -> None:
    st.markdown(
        f"""
        <div style="background:#111;border:1px solid #333;border-radius:10px;padding:10px 12px;margin-bottom:6px;">
          <div style="color:#bbb;font-size:12px;margin-bottom:4px;">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:700;">{value}</div>
          {f'<div style="color:#777;font-size:11px;margin-top:3px;">{help_text}</div>' if help_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    view = df.tail(25).iloc[::-1].copy()
    original_cols = set(view.columns)
    index = view.index

    if 'datetime_paris' in view.columns:
        heure_series = view['datetime_paris'].dt.strftime('%d/%m %H:%M:%S')
    elif 'datetime_utc' in view.columns:
        heure_series = pd.to_datetime(view['datetime_utc'], utc=True, errors='coerce').dt.tz_convert(TZ_PARIS).dt.strftime('%d/%m %H:%M:%S')
    else:
        heure_series = pd.Series([''] * len(view), index=index)

    prob_sources = {
        'bullish': ['prob_bull', 'prob_pos', 'prob_long'],
        'neutral': ['prob_neut', 'prob_neutral'],
        'bearish': ['prob_bear', 'prob_neg', 'prob_short'],
    }
    prob_numeric: Dict[str, pd.Series] = {}
    for label, candidates in prob_sources.items():
        series = None
        for candidate in candidates:
            if candidate in view.columns:
                series = pd.to_numeric(view[candidate], errors='coerce')
                break
        if series is None:
            series = pd.Series(np.nan, index=index)
        prob_numeric[label] = series

    conf_series = pd.Series(np.nan, index=index)
    if 'confidence' in original_cols:
        conf_series = pd.to_numeric(view['confidence'], errors='coerce')
    if conf_series.isna().all() and prob_numeric:
        prob_df = pd.DataFrame(prob_numeric)
        conf_series = prob_df.max(axis=1)
    conf_pct = conf_series.clip(lower=0.0, upper=1.0).mul(100)
    confidence_display = conf_pct.map(lambda x: f"{x:.1f} %" if pd.notna(x) else 'n/a')

    prediction_series = view['prediction'] if 'prediction' in view.columns else pd.Series([''] * len(view), index=index)
    signal_display = prediction_series.map(lambda x: LABEL_TRANSLATIONS.get(str(x).lower(), _clean_text(x) or '—'))

    ret_series = pd.Series(np.nan, index=index)
    if 'ret_pred' in original_cols:
        ret_series = pd.to_numeric(view['ret_pred'], errors='coerce')
    ret_display = ret_series.mul(100).map(lambda x: f"{x:+.2f} %" if pd.notna(x) else 'n/a')

    mag_series = pd.Series(np.nan, index=index)
    if 'mag_pred' in original_cols:
        mag_series = pd.to_numeric(view['mag_pred'], errors='coerce')
    mag_display = mag_series.abs().mul(100).map(lambda x: f"{x:.2f} %" if pd.notna(x) else 'n/a')

    mag_bucket_series = view['mag_bucket'] if 'mag_bucket' in view.columns else pd.Series(['n/a'] * len(view), index=index)
    mag_bucket_display = mag_bucket_series.fillna('n/a')

    features_status_series = view['features_status'] if 'features_status' in view.columns else pd.Series([''] * len(view), index=index)
    features_status_display = features_status_series.map(lambda x: _clean_text(x) or 'n/a')

    article_status_series = view['article_status'] if 'article_status' in view.columns else pd.Series([''] * len(view), index=index)
    article_status_display = article_status_series.map(lambda x: _clean_text(x) or 'n/a')

    article_found_series = view['article_found'] if 'article_found' in view.columns else pd.Series([None] * len(view), index=index)
    article_found_display = article_found_series.map(lambda x: '✅' if x is True else ('❌' if x is False else '?'))

    article_chars_series = view['article_chars'] if 'article_chars' in view.columns else pd.Series([0] * len(view), index=index)
    article_chars_display = pd.to_numeric(article_chars_series, errors='coerce').fillna(0).astype(int)

    text_source_series = view['text_source'] if 'text_source' in view.columns else pd.Series([''] * len(view), index=index)
    text_source_map = {'title': 'titre', 'summary': 'résumé', 'article': 'article'}
    text_source_display = text_source_series.map(lambda x: text_source_map.get(str(x).lower(), _clean_text(x) or 'n/a'))

    title_series = view['title'] if 'title' in view.columns else pd.Series([''] * len(view), index=index)
    title_display = title_series.map(_clean_text)

    prob_bull_display = prob_numeric['bullish'].clip(lower=0.0, upper=1.0).mul(100).map(lambda x: f"{x:.1f} %" if pd.notna(x) else 'n/a')
    prob_neut_display = prob_numeric['neutral'].clip(lower=0.0, upper=1.0).mul(100).map(lambda x: f"{x:.1f} %" if pd.notna(x) else 'n/a')
    prob_bear_display = prob_numeric['bearish'].clip(lower=0.0, upper=1.0).mul(100).map(lambda x: f"{x:.1f} %" if pd.notna(x) else 'n/a')

    table = pd.DataFrame(
        {
            'Heure (Paris)': heure_series,
            'Signal': signal_display,
            'Confiance': confidence_display,
            'Retour 60 min': ret_display,
            'Prob. haussière': prob_bull_display,
            'Prob. neutre': prob_neut_display,
            'Prob. baissière': prob_bear_display,
            'Amplitude 60 min': mag_display,
            'Catégorie amplitude': mag_bucket_display,
            'Statut features': features_status_display,
            'Statut article': article_status_display,
            'Article récupéré': article_found_display,
            'Taille article': article_chars_display,
            'Source texte': text_source_display,
            'Titre': title_display,
        }
    )
    return table


def style_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    return df


st.set_page_config(page_title="BTC Live Bot", layout="wide")
st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")
st.title("BTC Live Bot - Paper Trading (Streamlit)")

state = load_state()
preds_df = load_predictions()

status_cols = st.columns(3)
rss_ok, rss_detail = check_rss_status()
model_ok, model_detail = check_model_status(preds_df)
trader_ok, trader_detail = check_trader_status(state)
render_status_badge(status_cols[0], "Flux RSS", rss_ok, rss_detail)
model_label = f"Modèle {MODEL_DIR.name}"
render_status_badge(status_cols[1], model_label, model_ok, model_detail)
render_status_badge(status_cols[2], "Trader fictif", trader_ok, trader_detail)

with st.sidebar:
    st.header("Model info")
    info: Dict[str, Any] = {}
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        try:
            info = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"Unable to read config.json: {exc}")
    trainer_path = MODEL_DIR / "trainer_state.json"
    if trainer_path.exists():
        try:
            trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
            info["best_metric"] = trainer.get("best_metric")
            info["best_model_checkpoint"] = trainer.get("best_model_checkpoint")
        except Exception as exc:
            st.warning(f"Unable to read trainer_state.json: {exc}")
    if info:
        st.json(info)
    else:
        st.info("No model metadata found.")

    if st.button("Reset session state"):
        if STATE_FILE.exists():
            STATE_FILE.unlink(missing_ok=True)
        state = load_state()
        st.success("bot_state.json has been reset.")

st.caption(f"Dernière actualisation (Paris) : **{now_paris_str()}** – rafraîchissement automatique toutes les {REFRESH_SEC} s")

col_left, col_right = st.columns([1.6, 1.4], gap="large")

with col_left:
    st.subheader("Graphique BTC/USDT")
    interval = st.selectbox("Intervalle", INTERVALS, index=0)
    try:
        df_klines = fetch_klines(interval=interval, limit=400 if interval in ("1m", "5m") else 300)
    except Exception as exc:
        st.error(f"Failed to fetch Binance klines: {exc}")
        df_klines = pd.DataFrame()

    ticker = fetch_btc_ticker()
    price = ticker.get("price") if isinstance(ticker, dict) else None

    if not df_klines.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_klines["open_time"],
                    open=df_klines["open"],
                    high=df_klines["high"],
                    low=df_klines["low"],
                    close=df_klines["close"],
                    name="BTCUSDT",
                )
            ]
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            uirevision="chart_lock",
        )
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    else:
        st.info("En attente de données marché...")

with col_right:
    st.subheader("Instantané marché")
    if isinstance(ticker, dict) and ticker.get("price") is not None:
        pct = ticker.get("change_pct") or 0.0
        color = "#16c784" if pct >= 0 else "#ea3943"
        kpi_block("Prix (USD)", f"{ticker['price']:,.2f} $", color=color, help_text=f"Plus haut {ticker['high']:.0f} / Plus bas {ticker['low']:.0f}")
        price_eur = _as_float(ticker.get("price_eur"))
        if price_eur is not None:
            pct_eur = _as_float(ticker.get("change_pct_eur")) or 0.0
            color_eur = "#16c784" if pct_eur >= 0 else "#ea3943"
            high_eur = _as_float(ticker.get("high_eur"))
            low_eur = _as_float(ticker.get("low_eur"))
            help_eur = None
            if high_eur is not None and low_eur is not None:
                help_eur = f"Plus haut {high_eur:,.0f} / Plus bas {low_eur:,.0f}"
            kpi_block("Prix (EUR)", f"{price_eur:,.2f} €", color=color_eur, help_text=help_eur)
        kpi_block("Variation 24 h", f"{pct:+.2f} %")
        kpi_block("Volume 24 h", f"{ticker['volume']:.2f}")
    else:
        st.error(f"Impossible de récupérer le ticker : {ticker.get('error') if isinstance(ticker, dict) else 'erreur inconnue'}")

    st.subheader("Dernier signal")
    last_signal = resolve_last_signal(preds_df, state)
    if last_signal:
        pred_label = _normalize_label(last_signal.get('prediction'))
        raw_label = _clean_text(last_signal.get('prediction'))
        label_text = LABEL_TRANSLATIONS.get(pred_label, raw_label or '—')

        icon_map = {'bullish': '\U0001F680', 'bearish': '\U0001F53B', 'neutral': '\u2696\ufe0f'}
        icon = icon_map.get(pred_label, '\u2139\ufe0f')

        conf_val = _as_float(last_signal.get('confidence'))
        conf_display = 'n/a'
        if conf_val is not None:
            conf_display = f"{max(0.0, min(conf_val, 1.0)) * 100:.1f} %"

        ret_val = _as_float(last_signal.get('ret_pred'))
        mag_val = _as_float(last_signal.get('mag_pred'))
        if mag_val is None and ret_val is not None:
            mag_val = abs(ret_val)
        leverage = _as_int(last_signal.get('planned_leverage'))
        risk_fraction = _as_float(last_signal.get('risk_fraction'))

        if leverage is None:
            trades = state.get('trades') or []
            if trades:
                leverage = _as_int(trades[-1].get('leverage'))
        if risk_fraction is None:
            trades = state.get('trades') or []
            if trades:
                risk_fraction = _as_float(trades[-1].get('risk_fraction'))

        url_val = _clean_text(last_signal.get('url'))
        has_url = bool(url_val)

        article_found = _as_bool(last_signal.get('article_found'))
        raw_status = _clean_text(last_signal.get('article_status'))
        if article_found is True:
            article_mark = '✅'
            article_status = raw_status or 'article disponible'
        elif article_found is False and has_url:
            article_mark = '🔗'
            article_status = raw_status or 'URL disponible'
        elif article_found is False:
            article_mark = '❌'
            article_status = raw_status or 'non disponible'
        else:
            article_mark = '❓'
            article_status = raw_status or ('URL disponible' if has_url else 'non disponible')
        article_status = article_status.capitalize()

        article_chars = _as_int(last_signal.get('article_chars'))
        text_source = _clean_text(last_signal.get('text_source')) or ''
        text_source_map = {'title': 'titre', 'summary': 'résumé', 'article': 'article'}
        source_label = text_source_map.get(text_source.lower(), text_source) if text_source else ''
        features_status = _clean_text(last_signal.get('features_status')) or 'n/a'

        event_time = last_signal.get('event_time')
        event_str = None
        if isinstance(event_time, str) and event_time:
            try:
                event_time = pd.to_datetime(event_time, utc=True)
            except Exception:
                event_time = None
        if isinstance(event_time, pd.Timestamp):
            event_paris = event_time.tz_convert(TZ_PARIS) if event_time.tzinfo else event_time.tz_localize(TZ_PARIS)
            event_str = event_paris.strftime('%d/%m %H:%M:%S')

        lines = []
        if event_str:
            lines.append(f"Horodatage : {event_str}")
        signal_line = f"{icon} **{label_text}**"
        if conf_display != 'n/a':
            signal_line += f" - confiance {conf_display}"
        lines.append(signal_line)

        ret_source = last_signal.get('ret_source')
        if ret_val is not None and ret_source == 'predictions':
            if mag_val is not None:
                lines.append(f"Retour 60 min : {ret_val * 100:+.2f} % (|{mag_val * 100:.2f} %|)")
            else:
                lines.append(f"Retour 60 min : {ret_val * 100:+.2f} %")
        else:
            lines.append("Retour 60 min : n/a")

        if leverage is not None:
            lines.append(f"Levier planifié : x{leverage}")
        else:
            lines.append("Levier planifié : n/a")

        if risk_fraction is not None:
            lines.append(f"Risque engagé : {risk_fraction * 100:.2f} %")
        else:
            lines.append("Risque engagé : n/a")

        article_line = f"Article : {article_mark} {article_status}"
        if source_label:
            article_line += f" via {source_label}"
        if article_chars and article_chars > 0:
            article_line += f" - {article_chars} car."
        lines.append(article_line)

        if features_status and features_status.lower() != 'n/a':
            lines.append(f"Statut features : {features_status}")

        st.markdown("\n".join(f"- {line}" for line in lines if line))

        title = _clean_text(last_signal.get('title'))
        if title:
            st.caption(title)
        url = _clean_text(last_signal.get('url'))
        if url:
            st.markdown(f"[Ouvrir l'article]({url})")
    else:
        st.info("En attente d'un signal du modèle.")
    st.subheader("Prédictions en direct")
    if not preds_df.empty:
        table_df = format_prediction_table(preds_df)
        st.dataframe(table_df, width="stretch", height=420)
    else:
        st.info("En attente de nouvelles lignes dans live_predictions.csv...")

bottom_left, bottom_right = st.columns([1.1, 1.3], gap="large")

with bottom_left:
    st.subheader("État du bot")
    equity = state.get("equity", 0.0)
    start_equity = state.get("starting_equity", 0.0)
    pnl_total = equity - start_equity
    pnl_pct = (pnl_total / start_equity * 100.0) if start_equity else 0.0

    position = state.get("position")
    if position:
        info = f"{position['side'].upper()} @ {position['entry']:.2f} $"
        if position.get("leverage"):
            info += f" — lev x{position['leverage']}"
        kpi_block("Position ouverte", info, help_text=f"TP {position.get('tp',0):.2f} / SL {position.get('sl',0):.2f}")
    else:
        kpi_block("Position ouverte", "Aucune")

    kpi_block("Capital", f"{equity:,.2f} $", help_text=f"Départ {start_equity:,.0f} $")
    kpi_block("PnL total", f"{pnl_total:+,.2f} $", color="#16c784" if pnl_total >= 0 else "#ea3943", help_text=f"{pnl_pct:+.2f} %")

    st.markdown("**Transactions récentes**")
    trades = state.get("trades", [])
    if trades:
        trades_df = pd.DataFrame(trades).tail(20).iloc[::-1]
        st.dataframe(trades_df, width="stretch", height=320)
    else:
        st.info("Aucune transaction enregistrée pour le moment.")

with bottom_right:
    st.subheader("Courbe d'équité")
    curve = state.get("equity_curve", [])
    if len(curve) >= 2:
        ts = []
        values = []
        for t, v in curve:
            dt = pd.to_datetime(t, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            ts.append(dt.tz_convert(TZ_PARIS))
            values.append(float(v))
        if ts:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=values, mode="lines", name="Equity"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", uirevision="equity_lock")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Courbe d'équité indisponible pour l'instant.")
    else:
        st.info("La courbe d'équité apparaîtra après quelques rafraîchissements.")

st.caption("Paper trading dashboard — not financial advice.")






