"""
v4 增强特征工程：在 v3 基础上增加分段余震统计、ETAS 近似、构造代理等。

特征维度: ~52 维（v3 的 34 维 + 18 维新特征）。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

WINDOW_CUTOFF = {"T1": 0, "T2": 24, "T3": 72}
WINDOW_BOUNDS = {"T1": (0, 24), "T2": (24, 72), "T3": (72, 168)}
MAG_TYPES = ["Mw", "Ms", "M", "mww", "mwc", "mb", "ml"]
SOURCES = ["USGS", "CENC"]
DEFAULT_OMORI = {"K": 1.0, "c": 0.05, "p": 1.1}
MC_DEFAULT = 3.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _omori_rate(t: np.ndarray, K: float, c: float, p: float) -> np.ndarray:
    return K / np.power(c + t, p)


def fit_omori(hours: np.ndarray) -> Dict[str, float]:
    if len(hours) < 5:
        return DEFAULT_OMORI.copy()
    max_h = min(float(hours.max()), 168.0)
    bins = np.linspace(0.1, max_h, min(20, max(8, int(max_h / 6))))
    counts, edges = np.histogram(hours, bins=bins)
    t_centers = (edges[:-1] + edges[1:]) / 2
    mask = counts > 0
    if mask.sum() < 3:
        return DEFAULT_OMORI.copy()
    try:
        popt, _ = curve_fit(
            _omori_rate, t_centers[mask], counts[mask].astype(float),
            p0=[1.0, 0.05, 1.1],
            bounds=([0.01, 0.001, 0.5], [100.0, 5.0, 2.5]),
            maxfev=5000,
        )
        return {"K": float(popt[0]), "c": float(popt[1]), "p": float(popt[2])}
    except Exception:
        return DEFAULT_OMORI.copy()


def estimate_b_value(mags: np.ndarray, mc: float = MC_DEFAULT) -> float:
    mags = mags[mags >= mc]
    if len(mags) < 5:
        return 1.0
    bins = np.arange(mc, mags.max() + 0.1, 0.1)
    if len(bins) < 2:
        return 1.0
    counts, _ = np.histogram(mags, bins=bins)
    cum = np.cumsum(counts[::-1])[::-1]
    m_bins = bins[:-1] + 0.05
    mask = cum > 0
    if mask.sum() < 3:
        return 1.0
    slope, _ = np.polyfit(m_bins[mask], np.log10(cum[mask]), 1)
    b = -slope * math.log(10)
    return float(np.clip(b, 0.5, 1.5))


def bath_mag_pred(mainshock_mag: float, depth: float, t_start: float, t_end: float) -> float:
    if mainshock_mag >= 8.0:
        delta_m = 1.4
    elif mainshock_mag >= 7.5:
        delta_m = 1.3
    elif mainshock_mag >= 7.0:
        delta_m = 1.2
    elif mainshock_mag >= 6.5:
        delta_m = 1.1
    else:
        delta_m = 1.0
    if depth and not np.isnan(depth) and depth > 50:
        delta_m += 0.1
    if t_start == 0:
        factor = 0.95
    elif t_start == 24:
        factor = 0.90
    else:
        factor = 0.85
    return (mainshock_mag - delta_m) * factor


def omori_time_pred(mainshock_mag: float, t_start: float, t_end: float) -> float:
    if t_start == 0:
        h = 1.0 + 3.0 * (mainshock_mag - 6.0) / 3.0
        return float(np.clip(h, 1.0, min(12.0, t_end - 0.5)))
    if t_start == 24:
        h = 24.0 + 8.0 * (mainshock_mag - 6.0) / 3.0
        return float(np.clip(h, 24.0, min(48.0, t_end - 0.5)))
    h = 72.0 + 15.0 * (mainshock_mag - 6.0) / 3.0
    return float(np.clip(h, 72.0, min(120.0, t_end - 0.5)))


def gr_upper_bound(mainshock_mag: float, b_value: float, n_events: int) -> float:
    if n_events < 3:
        return mainshock_mag - 1.0
    a = math.log10(max(n_events, 1)) + b_value * MC_DEFAULT
    m_upper = a / b_value if b_value > 0 else mainshock_mag - 1.0
    return float(min(mainshock_mag - 0.3, max(MC_DEFAULT, m_upper)))


# ---------- v4 new feature helpers ----------

def _segment_counts(hours: np.ndarray, mags: np.ndarray, cutoff: float):
    """分段统计: 0-1h, 1-6h, 6-12h, 12-24h, 24-48h, 48-72h (截止前可见部分)。"""
    segs = [(0, 1), (1, 6), (6, 12), (12, 24), (24, 48), (48, 72)]
    counts = []
    max_mags = []
    for lo, hi in segs:
        if lo >= cutoff:
            counts.append(0)
            max_mags.append(0.0)
            continue
        hi_eff = min(hi, cutoff)
        mask = (hours >= lo) & (hours < hi_eff)
        counts.append(int(mask.sum()))
        max_mags.append(float(mags[mask].max()) if mask.any() else 0.0)
    return counts, max_mags


def _interevent_features(hours: np.ndarray):
    """余震间隔时间统计: ETAS 近似。"""
    if len(hours) < 2:
        return 0.0, 0.0, 0.0
    sorted_h = np.sort(hours)
    intervals = np.diff(sorted_h)
    if len(intervals) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(intervals)),
        float(np.std(intervals)) if len(intervals) > 1 else 0.0,
        float(np.median(intervals)),
    )


def _mag_distribution_features(mags: np.ndarray, mainshock_mag: float):
    """震级分布高级特征。"""
    if len(mags) == 0:
        return 0.0, 0.0, 0.0, 0.0
    m4_ratio = float((mags >= 4.0).sum()) / max(len(mags), 1)
    m5_ratio = float((mags >= 5.0).sum()) / max(len(mags), 1)
    mag_range = float(mags.max() - mags.min()) if len(mags) > 1 else 0.0
    delta_max = mainshock_mag - float(mags.max())
    return m4_ratio, m5_ratio, mag_range, delta_max


def _tectonic_proxy(lat: float, lon: float, depth: float) -> Tuple[float, float, float]:
    """构造环境代理特征 (无需外部数据)。"""
    is_subduction = 1.0 if (
        (abs(lat) < 50 and -180 <= lon <= -60) or  # Americas Pacific
        (0 < lat < 50 and 100 <= lon <= 180) or    # W. Pacific
        (-15 < lat < 15 and 90 <= lon <= 140) or   # Indonesia
        (30 < lat < 45 and 20 <= lon <= 50)         # Mediterranean
    ) else 0.0
    is_mid_ocean = 1.0 if (
        (abs(lat) > 50 and depth < 20) or
        (-60 < lat < -30 and -30 < lon < 60 and depth < 20)
    ) else 0.0
    is_continental = 1.0 if (
        (25 < lat < 45 and 60 <= lon <= 110) or    # Himalaya / Central Asia
        (35 < lat < 42 and 20 <= lon <= 45)         # Turkey-Iran
    ) else 0.0
    return is_subduction, is_mid_ocean, is_continental


def build_features(
    mainshock_mag: float,
    mainshock_depth: float,
    mainshock_lon: float,
    mainshock_lat: float,
    mag_type: str,
    source: str,
    aftershocks: pd.DataFrame,
    window: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """v4 特征向量 + 统计基线。"""
    cutoff = WINDOW_CUTOFF[window]
    t_start, t_end = WINDOW_BOUNDS[window]

    depth = float(mainshock_depth) if not (
        mainshock_depth is None or (isinstance(mainshock_depth, float) and np.isnan(mainshock_depth))
    ) else 20.0

    early = aftershocks[aftershocks["hours_after"] < cutoff].copy() if cutoff > 0 else aftershocks.iloc[0:0].copy()
    n_as = len(early)
    mags = early["Mag"].values if n_as > 0 else np.array([])
    hours = early["hours_after"].values if n_as > 0 else np.array([])

    # --- v3 features ---
    n_m5 = int((mags >= 5.0).sum()) if n_as else 0
    n_m4 = int((mags >= 4.0).sum()) if n_as else 0
    max_mag_early = float(mags.max()) if n_as else 0.0
    mean_mag = float(mags.mean()) if n_as else 0.0
    std_mag = float(mags.std()) if n_as > 1 else 0.0
    b_value = estimate_b_value(mags) if n_as else 1.0

    n_0_6 = int(((hours >= 0) & (hours < 6)).sum()) if n_as else 0
    n_6_24 = int(((hours >= 6) & (hours < 24)).sum()) if n_as and cutoff > 6 else 0
    rate_0_6 = n_0_6 / 6.0
    rate_6_24 = n_6_24 / 18.0 if cutoff > 6 else 0.0
    cum_energy = float(np.sum(np.power(10.0, 1.5 * mags))) if n_as else 0.0

    omori = fit_omori(hours) if n_as >= 5 else DEFAULT_OMORI.copy()

    if n_as > 0:
        dists = [
            haversine_km(mainshock_lon, mainshock_lat, row["Lon"], row["Lat"])
            for _, row in early.iterrows()
        ]
        mean_dist = float(np.mean(dists))
        spread_dist = float(np.std(dists)) if len(dists) > 1 else 0.0
    else:
        mean_dist = spread_dist = 0.0

    bath = bath_mag_pred(mainshock_mag, depth, t_start, t_end)
    omori_t = omori_time_pred(mainshock_mag, t_start, t_end)
    gr_ub = gr_upper_bound(mainshock_mag, b_value, n_as)

    mag_oh = [1.0 if mag_type == mt else 0.0 for mt in MAG_TYPES]
    src_oh = [1.0 if source == s else 0.0 for s in SOURCES]

    abs_lat = abs(mainshock_lat)
    shallow = 1.0 if depth < 30 else 0.0
    deep = 1.0 if depth > 50 else 0.0
    high_lat = 1.0 if abs_lat > 45 else 0.0

    # --- v4 new features ---
    seg_counts, seg_max_mags = _segment_counts(hours, mags, cutoff)
    iet_mean, iet_std, iet_median = _interevent_features(hours)
    m4_ratio, m5_ratio, mag_range, delta_max = _mag_distribution_features(mags, mainshock_mag)
    is_sub, is_mor, is_cont = _tectonic_proxy(mainshock_lat, mainshock_lon, depth)
    log_energy_main = math.log10(10.0 ** (1.5 * mainshock_mag))
    log_n_as = math.log10(n_as + 1)

    feat = [
        # v3 core (34 dims)
        mainshock_mag, depth, mainshock_lon, mainshock_lat, abs_lat,
        shallow, deep, high_lat,
        *mag_oh, *src_oh,
        n_as, n_m5, n_m4, max_mag_early, mean_mag, std_mag, b_value,
        rate_0_6, rate_6_24, math.log10(cum_energy + 1.0),
        omori["K"], omori["c"], omori["p"],
        mean_dist, spread_dist,
        t_start, t_end, t_end - t_start,
        bath, omori_t, gr_ub,
        # v4 additions (~18 dims)
        *seg_counts,        # 6 segment counts
        iet_mean, iet_std, iet_median,
        m4_ratio, m5_ratio, mag_range, delta_max,
        is_sub, is_mor, is_cont,
        log_energy_main, log_n_as,
    ]

    stat = {"bath_mag": bath, "omori_time": omori_t, "gr_upper": gr_ub}
    return np.array(feat, dtype=float), stat


FEATURE_NAMES = [
    # v3 (34)
    "M_main", "depth", "lon", "lat", "abs_lat", "shallow", "deep", "high_lat",
    *[f"mag_{m}" for m in MAG_TYPES],
    *[f"src_{s}" for s in SOURCES],
    "n_as", "n_M5", "n_M4", "max_mag_early", "mean_mag", "std_mag", "b_value",
    "rate_0_6h", "rate_6_24h", "log_cum_energy",
    "omori_K", "omori_c", "omori_p", "mean_dist_km", "spread_km",
    "t_start", "t_end", "window_hours",
    "bath_mag_pred", "omori_time_pred", "gr_upper_bound",
    # v4 new (18)
    "seg_0_1h", "seg_1_6h", "seg_6_12h", "seg_12_24h", "seg_24_48h", "seg_48_72h",
    "iet_mean", "iet_std", "iet_median",
    "m4_ratio", "m5_ratio", "mag_range", "delta_max_main",
    "is_subduction", "is_mid_ocean", "is_continental",
    "log_energy_main", "log_n_as",
]
