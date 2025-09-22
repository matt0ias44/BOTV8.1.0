#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live trader (paper trading).
- Watches live_predictions.csv (produced by bridge_inference.py)
- Reads latest BTC price from price_pipe.csv or Binance fallback
- Converts predictions into trades with dynamic sizing and risk settings
- Persists state in bot_state.json (consumed by Streamlit dashboard)
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

PRED_FILE = Path("live_predictions.csv")
PRICE_FILE = Path("price_pipe.csv")
STATE_FILE = Path("bot_state.json")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models/bert_v7_1_plus"))
REFRESH_SEC = 2

CONF_OPEN = 0.62
CONF_CLOSE = 0.55
CONF_BASELINE = 0.50
CONF_NEUTRAL_EXIT = 0.84
HOLD_NEUTRAL_MIN_SEC = 600
NEUTRAL_EXIT_MIN_STREAK = 2
NEUTRAL_EXIT_MAX_PROFIT_PCT = 0.0015
MIN_COOLDOWN_SEC = 60
BASE_POSITION_USD = 1000
MAX_LEVERAGE = 8
RISK_CAP_PCT = 0.02  # 2% de l'équité en risque par trade
MIN_RISK_SCORE = 0.12
ATR_VERY_LOW = 0.0015
ATR_LOW = 0.0030
ATR_HIGH = 0.0065
SL_FLOOR = 0.003  # 0.3%
TP_MULTIPLIER = 1.6
SL_MULTIPLIER = 0.85

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

LABELS = ["bearish", "neutral", "bullish"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def position_age_seconds(position: Optional[Dict]) -> Optional[float]:
    if not position:
        return None
    entry_ts_epoch = position.get("entry_ts_epoch")
    if entry_ts_epoch is not None:
        try:
            return max(0.0, time.time() - float(entry_ts_epoch))
        except (TypeError, ValueError):
            pass
    entry_ts = position.get("entry_ts")
    if not entry_ts:
        return None
    try:
        dt = datetime.fromisoformat(entry_ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, time.time() - dt.timestamp())


def safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_risk_fraction(confidence: float, mag_value: float, atr_pct: Optional[float],
                          thresholds: Optional[Dict[str, Dict[str, float]]]) -> float:
    edge = max(0.0, confidence - CONF_BASELINE)
    conf_weight = min(1.0, edge / max(1e-6, 1.0 - CONF_BASELINE))

    amp_weight = 0.4
    if thresholds and "60" in thresholds:
        bucket = thresholds["60"]
        q1 = bucket.get("q1", 0.001)
        q2 = bucket.get("q2", 0.003)
        if mag_value >= q2:
            amp_weight = 1.0
        elif mag_value >= q1:
            amp_weight = 0.65
        else:
            amp_weight = 0.35
    else:
        amp_weight = min(1.0, mag_value / 0.01)

    score = conf_weight * amp_weight
    if atr_pct is not None:
        if atr_pct < ATR_VERY_LOW:
            score *= 1.2
        elif atr_pct > ATR_HIGH:
            score *= 0.6
        elif atr_pct > ATR_LOW:
            score *= 0.85
    return max(0.0, min(1.0, score))


def select_leverage(mag_value: float, thresholds: Optional[Dict[str, Dict[str, float]]],
                    atr_pct: Optional[float]) -> int:
    base = 1
    if thresholds and "60" in thresholds:
        bucket = thresholds["60"]
        q1 = bucket.get("q1", 0.001)
        q2 = bucket.get("q2", 0.003)
        if mag_value >= q2:
            base = 4
        elif mag_value >= q1:
            base = 2
        else:
            base = 1
    else:
        base = 1 + int(min(MAX_LEVERAGE - 1, max(0.0, mag_value * 400)))

    if atr_pct is not None:
        if atr_pct >= ATR_HIGH:
            base = max(1, base - 2)
        elif atr_pct >= ATR_LOW:
            base = max(1, base - 1)
        elif atr_pct <= ATR_VERY_LOW:
            base = min(MAX_LEVERAGE, base + 1)
    return max(1, min(MAX_LEVERAGE, base))


def compute_tp_sl(ret_pred: float, confidence: float, atr_pct: Optional[float], side: str) -> Dict[str, float]:
    magnitude = abs(ret_pred)
    atr_component = atr_pct if atr_pct is not None else SL_FLOOR
    anchor = max(SL_FLOOR, max(magnitude, atr_component))
    conf_boost = 1.0 + max(0.0, confidence - CONF_BASELINE) * 1.2
    tp_pct = anchor * TP_MULTIPLIER * conf_boost
    sl_pct = max(SL_FLOOR, anchor * SL_MULTIPLIER / conf_boost)
    return {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    }


def build_risk_plan(state: Dict, confidence: float, ret_pred: float,
                    thresholds: Optional[Dict[str, Dict[str, float]]], atr_pct: Optional[float]) -> Optional[Dict]:
    mag_value = abs(ret_pred)
    risk_fraction = compute_risk_fraction(confidence, mag_value, atr_pct, thresholds)
    if risk_fraction < MIN_RISK_SCORE:
        return None

    leverage = select_leverage(mag_value, thresholds, atr_pct)
    equity = float(state.get("equity") or state.get("starting_equity") or 10000.0)
    equity = max(equity, float(state.get("starting_equity", 10000.0)))
    risk_budget = equity * RISK_CAP_PCT * risk_fraction
    if risk_budget <= 0:
        risk_budget = BASE_POSITION_USD * max(risk_fraction, 0.25)

    size_usd = risk_budget * leverage
    return {
        "size_usd": float(max(size_usd, BASE_POSITION_USD * 0.5)),
        "leverage": int(leverage),
        "risk_fraction": float(risk_fraction),
        "risk_budget": float(risk_budget),
        "confidence": float(confidence),
        "ret_pred": float(ret_pred),
        "mag_pred": float(mag_value),
        "atr_pct": None if atr_pct is None else float(atr_pct),
    }


def read_last_price() -> Optional[float]:
    if PRICE_FILE.exists():
        try:
            with PRICE_FILE.open("r", newline="", encoding="utf-8") as fh:
                last = None
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        try:
                            last = float(row[1])
                        except ValueError:
                            continue
                if last is not None:
                    return last
        except Exception as exc:
            print(f"[WARN] unable to read {PRICE_FILE}: {exc}")

    # fallback Binance
    try:
        resp = requests.get(BINANCE_PRICE_URL, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price"))
    except Exception as exc:
        print(f"[WARN] Binance price fetch failed: {exc}")
        return None


def init_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    data.setdefault("starting_equity", 10000.0)
                    data.setdefault("equity", data.get("starting_equity", 10000.0))
                    data.setdefault("position", None)
                    data.setdefault("trades", [])
                    data.setdefault("equity_curve", [[now_iso(), data.get("equity", 10000.0)]])
                    data.setdefault("last_pred_id", None)
                    data.setdefault("last_signal", None)
                    data.setdefault("strategy_settings", {})
                    return data
        except Exception as exc:
            print(f"[WARN] unable to read {STATE_FILE}: {exc}")
    return {
        "starting_equity": 10000.0,
        "equity": 10000.0,
        "position": None,
        "trades": [],
        "equity_curve": [[now_iso(), 10000.0]],
        "last_pred_id": None,
        "last_signal": None,
        "strategy_settings": {},
    }


def save_state(state: Dict) -> None:
    if len(state.get("equity_curve", [])) > 5000:
        state["equity_curve"] = state["equity_curve"][-5000:]
    if len(state.get("trades", [])) > 2000:
        state["trades"] = state["trades"][-2000:]
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            tmp.replace(STATE_FILE)
            break
        except (PermissionError, OSError) as exc:
            if attempt == max_attempts - 1:
                print(f"[WARN] unable to persist state after {max_attempts} attempts: {exc}")
                tmp.unlink(missing_ok=True)
            else:
                time.sleep(0.15)


def load_thresholds() -> Optional[Dict[str, Dict[str, float]]]:
    path = MODEL_DIR / "thresholds_mag.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[WARN] unable to read {path}: {exc}")
        return None


def latest_predictions(n: int = 40) -> pd.DataFrame:
    if not PRED_FILE.exists():
        return pd.DataFrame()
    try:
         df = pd.read_csv(PRED_FILE, engine="python", on_bad_lines="skip")
    except Exception as exc:
        print(f"[WARN] unable to read {PRED_FILE}: {exc}")
        return pd.DataFrame()
    required = {
        "news_id",
        "datetime_utc",
        "prediction",
        "prob_bear",
        "prob_neut",
        "prob_bull",
        "confidence",
        "ret_pred",
        "ret_30m_pred",
        "ret_120m_pred",
        "mag_pred",
        "mag_bucket",
        "mag_30m_pred",
        "mag_120m_pred",
        "feat_atr_60m_pct",
        "feat_atr_30m_pct",
        "feat_realized_vol_60m",
        "feat_realized_vol_60m_annual",
        "feat_vol_z_60m",
        "feat_volume_rate_30m",
        "article_status",
        "article_found",
        "article_chars",
        "text_source",
        "text_chars",
        "titles_joined",
        "body_concat",
    }
    for col in required:
        if col not in df.columns:
            df[col] = None
    return df.tail(n)


def close_position(state: Dict, price: float, reason: str = "CLOSE") -> None:
    pos = state.get("position")
    if not pos:
        return
    side = pos["side"]
    entry = float(pos["entry"])
    size = float(pos["size"])
    direction = -1.0 if side == "short" else 1.0
    pnl = direction * (price - entry) * (size / max(entry, 1e-9))
    state["equity"] = float(state["equity"]) + pnl
    state["trades"].append(
        {
            "time": now_iso(),
            "action": reason if reason.startswith("CLOSE") else f"CLOSE {side.upper()}",
            "price": round(price, 2),
            "size_usd": size,
            "pnl": round(pnl, 2),
            "title": pos.get("title", ""),
            "leverage": pos.get("leverage", 1),
            "risk_fraction": round(float(pos.get("risk_fraction", 0.0)), 4),
            "confidence": round(float(pos.get("confidence", 0.0)), 4),
            "ret_pred": round(float(pos.get("ret_pred", 0.0)), 6),
            "article_status": pos.get("article_status"),
        }
    )
    state["position"] = None


def open_position(state: Dict, side: str, price: float, title: str, plan: Dict) -> None:
    tp_sl = compute_tp_sl(plan.get("ret_pred", 0.0), plan.get("confidence", 0.0), plan.get("atr_pct"), side)
    tp = price * (1 + tp_sl["tp_pct"]) if side == "long" else price * (1 - tp_sl["tp_pct"])
    sl = price * (1 - tp_sl["sl_pct"]) if side == "long" else price * (1 + tp_sl["sl_pct"])
    size = plan.get("size_usd", BASE_POSITION_USD)
    leverage = plan.get("leverage", 1)
    state["position"] = {
        "side": side,
        "entry": float(price),
        "size": float(size),
        "entry_ts": now_iso(),
        "entry_ts_epoch": time.time(),
        "tp": float(tp),
        "sl": float(sl),
        "title": title[:160],
        "leverage": leverage,
        "mag_pred": plan.get("mag_pred", 0.0),
        "ret_pred": plan.get("ret_pred", 0.0),
        "risk_fraction": plan.get("risk_fraction", 0.0),
        "risk_budget": plan.get("risk_budget", 0.0),
        "atr_pct": plan.get("atr_pct"),
        "confidence": plan.get("confidence", 0.0),
        "article_status": plan.get("article_status"),
        "article_found": plan.get("article_found"),
        "text_source": plan.get("text_source"),
        "article_chars": plan.get("article_chars"),
        "text_chars": plan.get("text_chars"),
    }
    state["trades"].append(
        {
            "time": now_iso(),
            "action": f"OPEN {side.upper()}",
            "price": round(price, 2),
            "size_usd": float(size),
            "pnl": 0.0,
            "title": title[:160],
            "leverage": leverage,
            "risk_fraction": round(plan.get("risk_fraction", 0.0), 4),
            "confidence": round(plan.get("confidence", 0.0), 4),
            "ret_pred": round(plan.get("ret_pred", 0.0), 6),
            "article_status": plan.get("article_status"),
        }
    )


def mark_to_market(state: Dict, price: Optional[float]) -> None:
    if price is None:
        state["equity_curve"].append([now_iso(), state["equity"]])
        return
    pos = state.get("position")
    if pos:
        side = pos["side"]
        entry = pos["entry"]
        size = pos["size"]
        direction = -1.0 if side == "short" else 1.0
        unrealized = direction * (price - entry) * (size / max(entry, 1e-9))
        equity = state["starting_equity"] + unrealized
    else:
        equity = state["equity"]
    state["equity_curve"].append([now_iso(), equity])


def main():
    print("[LIVE] trader started")
    state = init_state()
    thresholds = load_thresholds()
    last_action_ts = 0.0
    seen_ids = set()
    neutral_streak = 0

    state.setdefault("strategy_settings", {})
    state["strategy_settings"].update(
        {
            "CONF_OPEN": CONF_OPEN,
            "CONF_CLOSE": CONF_CLOSE,
            "CONF_NEUTRAL_EXIT": CONF_NEUTRAL_EXIT,
            "HOLD_NEUTRAL_MIN_SEC": HOLD_NEUTRAL_MIN_SEC,
            "MIN_COOLDOWN_SEC": MIN_COOLDOWN_SEC,
            "NEUTRAL_EXIT_MIN_STREAK": NEUTRAL_EXIT_MIN_STREAK,
            "NEUTRAL_EXIT_MAX_PROFIT_PCT": NEUTRAL_EXIT_MAX_PROFIT_PCT,
        }
    )

    while True:
        time.sleep(REFRESH_SEC)
        price = read_last_price()

        if price and state.get("position"):
            pos = state["position"]
            side = pos["side"]
            hit_tp = (side == "long" and price >= pos["tp"]) or (side == "short" and price <= pos["tp"])
            hit_sl = (side == "long" and price <= pos["sl"]) or (side == "short" and price >= pos["sl"])
            if hit_tp or hit_sl:
                reason = "CLOSE TP" if hit_tp else "CLOSE SL"
                close_position(state, price, reason=reason)

        df = latest_predictions()
        if not df.empty:
            last_row = df.iloc[-1]
            news_id = str(last_row.get("news_id", ""))
            if news_id and news_id not in seen_ids:
                seen_ids.add(news_id)
                state["last_pred_id"] = news_id
                pred = str(last_row.get("prediction", "neutral")).lower()
                if pred == "neutral":
                    neutral_streak += 1
                else:
                    neutral_streak = 0
                confidence = float(last_row.get("confidence", 0.0) or 0.0)
                ret_pred = safe_float(last_row.get("ret_pred"))
                if ret_pred is None:
                    mag_raw = float(last_row.get("mag_pred", 0.0) or 0.0)
                    direction = -1.0 if pred == "bearish" else (1.0 if pred == "bullish" else 0.0)
                    ret_pred = mag_raw * direction
                mag_value = abs(ret_pred)
                atr_pct = safe_float(last_row.get("feat_atr_60m_pct"))
                title = str(last_row.get("title", ""))
                article_status = str(last_row.get("article_status", ""))
                article_found = bool(last_row.get("article_found", False))
                article_chars = int(float(last_row.get("article_chars", 0) or 0))
                text_source = str(last_row.get("text_source", ""))
                url = str(last_row.get("url", ""))

                text_chars = int(float(last_row.get("text_chars", 0) or 0))

                plan = build_risk_plan(state, confidence, ret_pred, thresholds, atr_pct)
                if plan:
                    plan.setdefault("article_status", article_status)
                    plan.setdefault("article_found", article_found)
                    plan.setdefault("text_source", text_source)
                    plan.setdefault("article_chars", article_chars)
                    plan.setdefault("text_chars", text_chars)

                signal_info = {
                    "time": now_iso(),
                    "news_id": news_id,
                    "prediction": pred,
                    "confidence": float(confidence),
                    "ret_pred": float(ret_pred),
                    "mag_pred": float(mag_value),
                    "atr_pct": None if atr_pct is None else float(atr_pct),
                    "article_status": article_status,
                    "article_found": article_found,
                    "article_chars": article_chars,
                    "text_chars": text_chars,
                    "text_source": text_source,
                    "title": title[:200],
                    "url": url,
                    "features_status": str(last_row.get("features_status", "")),
                    "position_age_sec": position_age_seconds(state.get("position")),
                    "neutral_exit_threshold": CONF_NEUTRAL_EXIT,
                    "neutral_exit_min_hold_sec": HOLD_NEUTRAL_MIN_SEC,
                    "neutral_exit_min_streak": NEUTRAL_EXIT_MIN_STREAK,
                    "neutral_exit_max_profit_pct": NEUTRAL_EXIT_MAX_PROFIT_PCT,
                    "neutral_streak": neutral_streak,
                }
                if plan:
                    signal_info.update(
                        {
                            "planned_leverage": plan.get("leverage"),
                            "risk_fraction": plan.get("risk_fraction"),
                            "risk_budget": plan.get("risk_budget"),
                        }
                    )
                state["last_signal"] = signal_info

                now_s = time.time()
                if now_s - last_action_ts >= MIN_COOLDOWN_SEC:
                    if state.get("position") is None:
                        if pred == "bullish" and confidence >= CONF_OPEN and price and plan:
                            open_position(state, "long", price, title, plan)
                            last_action_ts = now_s
                        elif pred == "bearish" and confidence >= CONF_OPEN and price and plan:
                            open_position(state, "short", price, title, plan)
                            last_action_ts = now_s
                    else:
                        pos = state["position"]
                        side = pos["side"]
                        if pred == "neutral":
                            hold_seconds = position_age_seconds(pos)
                            neutral_conf_ok = confidence >= CONF_NEUTRAL_EXIT
                            neutral_hold_ok = (
                                HOLD_NEUTRAL_MIN_SEC <= 0
                                or hold_seconds is None
                                or hold_seconds >= HOLD_NEUTRAL_MIN_SEC
                            )
                            pnl_pct = None
                            entry_price = None
                            if pos.get("entry") is not None:
                                try:
                                    entry_price = float(pos.get("entry"))
                                except (TypeError, ValueError):
                                    entry_price = None
                            if entry_price is None and price is not None:
                                entry_price = float(price)
                            if price is not None and entry_price is not None and entry_price != 0:
                                direction = -1.0 if side == "short" else 1.0
                                pnl_pct = direction * (price - entry_price) / max(entry_price, 1e-9)
                            neutral_pnl_ok = (
                                pnl_pct is None
                                or pnl_pct <= NEUTRAL_EXIT_MAX_PROFIT_PCT
                                or pnl_pct < 0
                            )
                            neutral_streak_ok = neutral_streak >= NEUTRAL_EXIT_MIN_STREAK
                            neutral_allowed = (
                                price is not None
                                and neutral_conf_ok
                                and neutral_hold_ok
                                and neutral_pnl_ok
                                and neutral_streak_ok
                            )
                            block_reason = None
                            if not neutral_conf_ok:
                                block_reason = "confidence"
                            elif not neutral_hold_ok:
                                block_reason = "min_hold"
                            elif not neutral_pnl_ok:
                                block_reason = "pnl"
                            elif not neutral_streak_ok:
                                block_reason = "streak"
                            if state.get("last_signal"):
                                state["last_signal"].update(
                                    {
                                        "neutral_exit_allowed": neutral_allowed,
                                        "neutral_conf_ok": neutral_conf_ok,
                                        "neutral_hold_ok": neutral_hold_ok,
                                        "neutral_pnl_ok": neutral_pnl_ok,
                                        "neutral_pnl_pct": pnl_pct,
                                        "neutral_streak_ok": neutral_streak_ok,
                                        "neutral_hold_seconds": hold_seconds,
                                        "neutral_exit_block_reason": block_reason,
                                        "position_age_sec": hold_seconds,
                                    }
                                )
                            if neutral_allowed:
                                close_position(state, price, reason="CLOSE neutral")
                                last_action_ts = now_s
                                neutral_streak = 0
                        elif side == "long" and pred == "bearish" and confidence >= CONF_OPEN and price:
                            close_position(state, price, reason="CLOSE flip")
                            last_action_ts = now_s
                            neutral_streak = 0
                            if plan:
                                open_position(state, "short", price, title, plan)
                        elif side == "short" and pred == "bullish" and confidence >= CONF_OPEN and price:
                            close_position(state, price, reason="CLOSE flip")
                            last_action_ts = now_s
                            neutral_streak = 0
                            if plan:
                                open_position(state, "long", price, title, plan)

        mark_to_market(state, price)
        save_state(state)


if __name__ == "__main__":
    main()
