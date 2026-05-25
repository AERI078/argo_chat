# data_pipeline/fetcher.py — DB-first fetch with ERDDAP fallback
# fetch order for every region/float request:
#   1. check if region is already in DB  → return immediately (fast)
#   2. if not → fetch from ERDDAP        → store in DB → return
# over time the DB fills up and ERDDAP is called less and less

import argopy
import pandas as pd
import threading
from config import DEFAULT_PRESSURE_RANGE
from data_pipeline.db import (
    is_region_cached, load_profiles_from_db, load_float_from_db,
    cache_profiles, log_fetched_region
)

argopy.set_options(src="erddap")

_NAMES = ["float_id", "lat", "lon", "date", "pressure_dbar", "temperature_c", "salinity_psu"]


# ── timeout helper ────────────────────────────────────────────────────────────

class FetchTimeoutError(Exception):
    pass


def _run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Runs func in a thread with a hard timeout. Windows-compatible."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise FetchTimeoutError(
            f"ERDDAP fetch timed out after {timeout_seconds}s. "
            "Try a smaller region or shorter date range."
        )
    if exception[0]:
        raise exception[0]
    return result[0]


# ── internal ERDDAP fetch ─────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types from raw argopy output."""
    if "TIME" in df.columns:
        cols = ["PLATFORM_NUMBER", "LATITUDE", "LONGITUDE", "TIME", "PRES", "TEMP", "PSAL"]
    else:
        cols = ["PLATFORM_NUMBER", "LATITUDE", "LONGITUDE", "JULD", "PRES", "TEMP", "PSAL"]
    df = df[cols].copy()
    df.columns = _NAMES
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["float_id"] = df["float_id"].astype(str)
    return df.dropna(subset=["temperature_c", "salinity_psu"])


def _erddap_fetch_region(lat_min, lat_max, lon_min, lon_max, date_start, date_end):
    """Raw ERDDAP fetch — no timeout, no DB interaction. Called via _run_with_timeout."""
    loader = argopy.DataFetcher(src="erddap", mode="standard").region(
        [lon_min, lon_max, lat_min, lat_max,
         DEFAULT_PRESSURE_RANGE[0], DEFAULT_PRESSURE_RANGE[1],
         date_start, date_end]
    )
    ds = loader.to_xarray()
    df = ds.to_dataframe()
    return _clean(df.reset_index())


def _erddap_fetch_float(float_id: str):
    """Raw ERDDAP float fetch — no timeout. Called via _run_with_timeout."""
    loader = argopy.DataFetcher(src="erddap", mode="standard").float(int(float_id))
    ds = loader.to_xarray()
    df = ds.to_dataframe()
    return _clean(df.reset_index())


# ── public API ────────────────────────────────────────────────────────────────

def fetch_indian_ocean_profiles() -> pd.DataFrame:
    """
    Fetches a focused sample for FAISS index building on cold start.
    DB-first: if we already have this region stored, skip ERDDAP entirely.
    """
    lat_min, lat_max = 5, 25
    lon_min, lon_max = 50, 75
    date_start, date_end = "2023-06", "2023-12"

    # check DB first — avoids ERDDAP call on every cold start after first run
    if is_region_cached(lat_min, lat_max, lon_min, lon_max, date_start, date_end):
        print("Startup fetch: loading from DB (skipping ERDDAP)...")
        df = load_profiles_from_db(lat_min, lat_max, lon_min, lon_max,
                                    date_start, date_end)
        if not df.empty:
            print(f"Startup fetch: loaded {len(df)} rows from DB.")
            return df
        print("DB returned empty for startup region — falling back to ERDDAP.")

    print("Startup fetch: hitting ERDDAP (first run for this region)...")
    df = _run_with_timeout(
        _erddap_fetch_region, 90,
        lat_min, lat_max, lon_min, lon_max, date_start, date_end
    )

    # store permanently so future startups skip ERDDAP
    cache_profiles(df)
    log_fetched_region(lat_min, lat_max, lon_min, lon_max,
                        date_start, date_end, row_count=len(df))
    print(f"Startup fetch: stored {len(df)} rows to DB.")
    return df


def fetch_profiles_by_region(lat_min: float, lat_max: float,
                              lon_min: float, lon_max: float,
                              date_start: str, date_end: str) -> pd.DataFrame:
    """
    Fetches Argo profiles for a region and date range.
    DB-first: returns stored data immediately if this region was previously fetched.
    Falls back to ERDDAP and stores the result for next time.
    """
    # ── DB-first ──────────────────────────────────────────────────────────────
    if is_region_cached(lat_min, lat_max, lon_min, lon_max, date_start, date_end):
        print(f"Region cache hit — loading from DB...")
        df = load_profiles_from_db(lat_min, lat_max, lon_min, lon_max,
                                    date_start, date_end)
        if not df.empty:
            print(f"Loaded {len(df)} rows from DB.")
            return df
        print("DB returned empty for cached region — falling back to ERDDAP.")

    # ── ERDDAP fallback ───────────────────────────────────────────────────────
    print(f"Region not in DB — fetching from ERDDAP: "
          f"lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] "
          f"{date_start}→{date_end}")
    df = _run_with_timeout(
        _erddap_fetch_region, 60,
        lat_min, lat_max, lon_min, lon_max, date_start, date_end
    )

    # store permanently
    cache_profiles(df)
    log_fetched_region(lat_min, lat_max, lon_min, lon_max,
                        date_start, date_end, row_count=len(df))
    print(f"Stored {len(df)} rows to DB.")
    return df


def fetch_float_by_id(float_id: str) -> pd.DataFrame:
    """
    Fetches full profile history for a specific Argo float.
    DB-first: returns stored data if this float has been fetched before.
    Falls back to ERDDAP and stores the result for next time.
    """
    # ── DB-first ──────────────────────────────────────────────────────────────
    df = load_float_from_db(float_id)
    if not df.empty:
        print(f"Float {float_id}: loaded {len(df)} rows from DB.")
        return df

    # ── ERDDAP fallback ───────────────────────────────────────────────────────
    print(f"Float {float_id} not in DB — fetching from ERDDAP...")
    df = _run_with_timeout(_erddap_fetch_float, 45, float_id)

    # store permanently
    cache_profiles(df)
    # log as a region covering the float's full extent
    if not df.empty:
        log_fetched_region(
            lat_min=float(df["lat"].min()), lat_max=float(df["lat"].max()),
            lon_min=float(df["lon"].min()), lon_max=float(df["lon"].max()),
            date_start=str(df["date"].min()), date_end=str(df["date"].max()),
            row_count=len(df)
        )
    print(f"Stored {len(df)} rows for float {float_id} to DB.")
    return df