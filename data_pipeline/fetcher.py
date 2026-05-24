# data_pipeline/fetcher.py — fetches Argo float data via argopy
# smaller default region and shorter date range to avoid ERDDAP timeouts

import argopy
import pandas as pd
from config import DEFAULT_PRESSURE_RANGE

argopy.set_options(src="erddap")

# columns we care about — same across all fetch functions
_COLS = ["PLATFORM_NUMBER", "LATITUDE", "LONGITUDE", "JULD", "PRES", "TEMP", "PSAL"]
_NAMES = ["float_id", "lat", "lon", "date", "pressure_dbar", "temperature_c", "salinity_psu"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[_COLS].copy()
    df.columns = _NAMES
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["float_id"] = df["float_id"].astype(str)
    return df.dropna(subset=["temperature_c", "salinity_psu"])


def fetch_indian_ocean_profiles() -> pd.DataFrame:
    """
    Fetches a small, focused sample for building the FAISS index at startup.
    Kept deliberately small — Arabian Sea, 6 months — to avoid ERDDAP timeouts.
    If you want more coverage, run this manually and save the FAISS index to disk.
    """
    loader = argopy.DataFetcher(src="erddap", mode="standard").region(
        [50, 75, 5, 25, 0, 100, "2023-06", "2023-12"]  # Arabian Sea, surface to 100 dbar
    )
    ds = loader.to_xarray()
    return _clean(ds.argo.point2dataframe())


def fetch_profiles_by_region(lat_min: float, lat_max: float,
                              lon_min: float, lon_max: float,
                              date_start: str, date_end: str) -> pd.DataFrame:
    """On-demand fetch for agent queries. Agent should pass focused regions."""
    loader = argopy.DataFetcher(src="erddap", mode="standard").region(
        [lon_min, lon_max, lat_min, lat_max,
         DEFAULT_PRESSURE_RANGE[0], DEFAULT_PRESSURE_RANGE[1],
         date_start, date_end]
    )
    ds = loader.to_xarray()
    return _clean(ds.argo.point2dataframe())


def fetch_float_by_id(float_id: str) -> pd.DataFrame:
    """Fetches full profile history for a specific Argo float."""
    loader = argopy.DataFetcher(src="erddap", mode="standard").float(int(float_id))
    ds = loader.to_xarray()
    return _clean(ds.argo.point2dataframe())