from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import signal

from .models import ChannelLag, PreprocessConfig, PreprocessResult
from .utils import apply_bz_effective_convention, compute_projection, reconstruct_signed_current_channels


OFFSET_CHANNELS = (
    "daq_input_v",
    "coil1_current_a",
    "coil2_current_a",
    "bx_mT",
    "by_mT",
    "bz_mT",
)

SMOOTHING_CHANNELS = (
    "daq_input_v",
    "coil1_current_a",
    "coil2_current_a",
    "coil1_current_signed_a",
    "coil2_current_signed_a",
    "i_sum",
    "i_diff",
    "i_custom",
    "i_sum_signed",
    "i_diff_signed",
    "i_custom_signed",
    "bx_mT",
    "by_mT",
    "bz_mT",
    "bmag_mT",
    "bproj_mT",
)


def apply_preprocessing(
    normalized_frame: pd.DataFrame,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Build a corrected dataset while keeping raw data untouched."""

    corrected = normalized_frame.copy()
    warnings: list[str] = []
    logs: list[str] = [f"전처리 설정: {asdict(config)}"]
    offsets: dict[str, float] = {}

    baseline_mask = pd.Series(False, index=corrected.index)
    if config.baseline_seconds > 0 and "time_s" in corrected.columns:
        baseline_mask = corrected["time_s"] <= config.baseline_seconds
        if baseline_mask.sum() < 3:
            warnings.append("baseline 구간 샘플 수가 부족하여 offset 제거를 건너뜁니다.")
        else:
            for channel in OFFSET_CHANNELS:
                if channel not in corrected.columns:
                    continue
                offset = float(pd.to_numeric(corrected.loc[baseline_mask, channel], errors="coerce").mean())
                if np.isnan(offset):
                    continue
                corrected[channel] = pd.to_numeric(corrected[channel], errors="coerce") - offset
                offsets[channel] = offset
            logs.append(f"offset 제거 채널 수: {len(offsets)}")

    for channel, multiplier in config.sign_flips.items():
        if channel not in corrected.columns or multiplier not in {-1, 1}:
            continue
        corrected[channel] = pd.to_numeric(corrected[channel], errors="coerce") * multiplier
        logs.append(f"{channel}: sign flip {multiplier}")

    signed_current_info = _recompute_derived_columns(corrected, config)
    if signed_current_info["reconstructed_columns"]:
        logs.append(
            "signed current reconstruction: "
            f"{', '.join(signed_current_info['reconstructed_columns'])} via {signed_current_info['reference_channel']}"
        )

    if config.smoothing_method != "none":
        corrected = _apply_smoothing(corrected, config)
        signed_current_info = _recompute_derived_columns(corrected, config)
        logs.append(f"smoothing 적용: {config.smoothing_method}")
        if signed_current_info["reconstructed_columns"]:
            logs.append(
                "signed current reconstruction: "
                f"{', '.join(signed_current_info['reconstructed_columns'])} via {signed_current_info['reference_channel']}"
            )

    lags: list[ChannelLag] = []
    if config.alignment_reference and config.alignment_targets:
        lags = estimate_channel_lags(
            frame=corrected,
            reference_channel=config.alignment_reference,
            target_channels=config.alignment_targets,
        )
        if config.apply_alignment:
            corrected = apply_channel_alignment(
                frame=corrected,
                lags=lags,
            )
            signed_current_info = _recompute_derived_columns(corrected, config)
            logs.append("추정된 시차를 corrected 데이터에 적용")
            if signed_current_info["reconstructed_columns"]:
                logs.append(
                    "signed current reconstruction: "
                    f"{', '.join(signed_current_info['reconstructed_columns'])} via {signed_current_info['reference_channel']}"
                )

    apply_bz_effective_convention(corrected)
    corrected = _apply_outlier_masks(corrected, config)
    return PreprocessResult(
        corrected_frame=corrected,
        offsets=offsets,
        lags=lags,
        warnings=warnings,
        logs=logs,
    )


def estimate_channel_lags(
    frame: pd.DataFrame,
    reference_channel: str,
    target_channels: tuple[str, ...],
) -> list[ChannelLag]:
    """Estimate channel lags from cross-correlation."""

    if reference_channel not in frame.columns or "time_s" not in frame.columns:
        return []

    lags: list[ChannelLag] = []
    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    dt = float(time_s.diff().median()) if time_s.notna().sum() > 1 else 0.0
    if dt <= 0:
        return []

    reference = pd.to_numeric(frame[reference_channel], errors="coerce")
    for channel in target_channels:
        if channel == reference_channel or channel not in frame.columns:
            continue
        lag = _estimate_single_lag(reference, pd.to_numeric(frame[channel], errors="coerce"), dt)
        if lag is not None:
            lags.append(ChannelLag(channel=channel, **lag))
    return lags


def apply_channel_alignment(frame: pd.DataFrame, lags: list[ChannelLag]) -> pd.DataFrame:
    """Shift aligned channels with interpolation onto the original time grid."""

    corrected = frame.copy()
    time = pd.to_numeric(corrected["time_s"], errors="coerce").to_numpy(dtype=float)
    for lag in lags:
        if lag.channel not in corrected.columns or abs(lag.lag_seconds) < 1e-12:
            continue
        values = pd.to_numeric(corrected[lag.channel], errors="coerce").to_numpy(dtype=float)
        shifted = np.interp(
            time,
            time + lag.lag_seconds,
            values,
            left=np.nan,
            right=np.nan,
        )
        corrected[lag.channel] = shifted
    return corrected


def _estimate_single_lag(
    reference: pd.Series,
    target: pd.Series,
    dt: float,
) -> dict[str, float | int] | None:
    valid = reference.notna() & target.notna()
    if valid.sum() < 8:
        return None

    ref = reference.loc[valid].to_numpy(dtype=float)
    tgt = target.loc[valid].to_numpy(dtype=float)
    ref = ref - ref.mean()
    tgt = tgt - tgt.mean()
    if np.allclose(ref.std(), 0.0) or np.allclose(tgt.std(), 0.0):
        return None

    corr = signal.correlate(ref, tgt, mode="full", method="fft")
    lag_axis = signal.correlation_lags(len(ref), len(tgt), mode="full")
    max_lag_samples = max(1, int(min(len(ref), len(tgt)) * 0.25))
    lag_mask = np.abs(lag_axis) <= max_lag_samples
    if not lag_mask.any():
        return None

    corr = corr[lag_mask]
    lag_axis = lag_axis[lag_mask]
    best_index = int(np.argmax(corr))
    lag_samples = int(lag_axis[best_index])
    denominator = np.linalg.norm(ref) * np.linalg.norm(tgt)
    correlation = float(corr[best_index] / denominator) if denominator else 0.0
    return {
        "lag_seconds": float(lag_samples * dt),
        "lag_samples": lag_samples,
        "correlation": correlation,
    }


def _apply_smoothing(frame: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    corrected = frame.copy()
    window = max(3, int(config.smoothing_window))
    if window % 2 == 0:
        window += 1

    for channel in SMOOTHING_CHANNELS:
        if channel not in corrected.columns:
            continue
        series = pd.to_numeric(corrected[channel], errors="coerce")
        if config.smoothing_method == "moving_average":
            corrected[channel] = series.rolling(window=window, center=True, min_periods=1).mean()
            continue

        if config.smoothing_method == "savitzky_golay":
            valid = series.notna()
            if valid.sum() < window:
                continue
            smoothed = series.copy()
            smoothed.loc[valid] = signal.savgol_filter(
                series.loc[valid].to_numpy(dtype=float),
                window_length=window,
                polyorder=min(config.savgol_polyorder, window - 1),
                mode="interp",
            )
            corrected[channel] = smoothed

    return corrected


def _apply_outlier_masks(frame: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    corrected = frame.copy()
    corrected["analysis_mask"] = True
    threshold = float(config.outlier_zscore_threshold)
    if threshold <= 0:
        return corrected

    mask_columns: list[str] = []
    for channel in SMOOTHING_CHANNELS:
        if channel not in corrected.columns:
            continue
        series = pd.to_numeric(corrected[channel], errors="coerce")
        std = float(series.std(ddof=0))
        if np.isnan(std) or std == 0:
            continue
        zscore = (series - float(series.mean())) / std
        mask = zscore.abs() <= threshold
        mask_column = f"mask_{channel}"
        corrected[mask_column] = mask.fillna(False)
        mask_columns.append(mask_column)

    if mask_columns:
        corrected["analysis_mask"] = corrected[mask_columns].all(axis=1)
    return corrected


def _recompute_derived_columns(frame: pd.DataFrame, config: PreprocessConfig) -> dict[str, object]:
    for channel in ("coil1_current_a", "coil2_current_a", "bx_mT", "by_mT", "bz_mT"):
        if channel in frame.columns:
            frame[channel] = pd.to_numeric(frame[channel], errors="coerce")

    frame["i_sum"] = frame[["coil1_current_a", "coil2_current_a"]].sum(axis=1, min_count=1)
    frame["i_diff"] = frame["coil1_current_a"] - frame["coil2_current_a"]
    frame["i_custom"] = (
        config.custom_current_alpha * frame["coil1_current_a"]
        + config.custom_current_beta * frame["coil2_current_a"]
    )
    signed_current_info = reconstruct_signed_current_channels(
        frame,
        custom_current_alpha=config.custom_current_alpha,
        custom_current_beta=config.custom_current_beta,
    )
    frame["bmag_mT"] = np.sqrt(
        np.square(frame["bx_mT"].fillna(0.0))
        + np.square(frame["by_mT"].fillna(0.0))
        + np.square(frame["bz_mT"].fillna(0.0))
    )
    frame["bproj_mT"] = compute_projection(
        bx=frame["bx_mT"].fillna(0.0),
        by=frame["by_mT"].fillna(0.0),
        bz=frame["bz_mT"].fillna(0.0),
        vector=config.projection_vector,
    )
    return signed_current_info
