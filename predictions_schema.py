"""Shared schema information for live_predictions.csv."""

from __future__ import annotations

EXPORTED_FEATURE_KEYS: tuple[str, ...] = (
    "feat_atr_30m_pct",
    "feat_atr_60m_pct",
    "feat_realized_vol_60m",
    "feat_realized_vol_60m_annual",
    "feat_vol_z_60m",
    "feat_volume_rate_30m",
)

PREDICTIONS_BASE_COLUMNS: tuple[str, ...] = (
    "news_id",
    "datetime_paris",
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
    "mag_30m_pred",
    "mag_120m_pred",
    "mag_bucket",
    "features_status",
    "title",
    "summary",
    "url",
    "source",
    "titles_joined",
    "body_concat",
    "text_source",
    "text_chars",
    "article_status",
    "article_found",
    "article_chars",
    "processed_at",
)


def predictions_header() -> list[str]:
    """Return the expected header for ``live_predictions.csv``."""

    return list(PREDICTIONS_BASE_COLUMNS + EXPORTED_FEATURE_KEYS)


PREDICTIONS_HEADER: list[str] = predictions_header()

