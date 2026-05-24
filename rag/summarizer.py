# rag/summarizer.py — converts raw float DataFrame rows into text summaries
# these summaries are what gets embedded and stored in FAISS

import pandas as pd


def summarize_profiles(df: pd.DataFrame) -> list[str]:
    """
    Groups measurements by float + date and produces one summary per profile.
    A profile = one float on one day at various depths.
    """
    summaries = []
    grouped = df.groupby(["float_id", "date"])

    for (float_id, date), group in grouped:
        lat = group["lat"].iloc[0]
        lon = group["lon"].iloc[0]
        avg_temp = group["temperature_c"].mean().round(2)
        avg_sal = group["salinity_psu"].mean().round(2)
        max_depth = group["pressure_dbar"].max().round(0)
        region = _region_label(lat, lon)

        summary = (
            f"Argo float {float_id} recorded a profile on {date} near {region} "
            f"(lat {lat:.2f}, lon {lon:.2f}). "
            f"Average temperature: {avg_temp}°C, average salinity: {avg_sal} PSU. "
            f"Profile depth: {max_depth} dbar."
        )
        summaries.append(summary)

    return summaries


def _region_label(lat: float, lon: float) -> str:
    """Simple rule-based label for major Indian Ocean regions."""
    if lat > 5 and lon > 60 and lon < 80:
        return "Arabian Sea"
    elif lat < 5 and lat > -5 and lon > 70:
        return "Equatorial Indian Ocean"
    elif lat < -5 and lat > -25:
        return "Southern Indian Ocean"
    elif lat > 5 and lon > 80:
        return "Bay of Bengal"
    else:
        return "Indian Ocean"
