# data_pipeline/fetcher.py — fetches Argo float data via argopy
# smaller default region and shorter date range to avoid ERDDAP timeouts

# data_pipeline/fetcher.py

import argopy
import pandas as pd
from config import DEFAULT_PRESSURE_RANGE

argopy.set_options(src="erddap")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Argo profile data.
    Column names changed in recent argopy versions - this handles both old and new.
    """
    # Print columns to debug (remove after first successful run)
    print("Available columns:", df.columns.tolist())
    
    # Map old column names to new ones if needed
    column_mapping = {
        'JULD': 'TIME',  # Time column renamed
        'PLATFORM_NUMBER': 'PLATFORM_NUMBER',
        'LATITUDE': 'LATITUDE', 
        'LONGITUDE': 'LONGITUDE',
        'PRES': 'PRES',
        'TEMP': 'TEMP',
        'PSAL': 'PSAL'
    }
    
    # Try new column names first, fall back to old
    if 'TIME' in df.columns:
        # New argopy format
        cols = ['PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE', 'TIME', 'PRES', 'TEMP', 'PSAL']
    else:
        # Old argopy format
        cols = ['PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE', 'JULD', 'PRES', 'TEMP', 'PSAL']
    
    df = df[cols].copy()
    df.columns = ["float_id", "lat", "lon", "date", "pressure_dbar", "temperature_c", "salinity_psu"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["float_id"] = df["float_id"].astype(str)
    return df.dropna(subset=["temperature_c", "salinity_psu"])


def fetch_indian_ocean_profiles() -> pd.DataFrame:
    loader = argopy.DataFetcher(src="erddap", mode="standard").region(
        [50, 75, 5, 25, 0, 100, "2023-06", "2023-12"]
    )
    ds = loader.to_xarray()
    df = ds.to_dataframe()
    return _clean(df.reset_index())


def fetch_profiles_by_region(lat_min: float, lat_max: float,
                              lon_min: float, lon_max: float,
                              date_start: str, date_end: str) -> pd.DataFrame:
    loader = argopy.DataFetcher(src="erddap", mode="standard").region(
        [lon_min, lon_max, lat_min, lat_max,
         DEFAULT_PRESSURE_RANGE[0], DEFAULT_PRESSURE_RANGE[1],
         date_start, date_end]
    )
    ds = loader.to_xarray()
    df = ds.to_dataframe()
    return _clean(df.reset_index())


def fetch_float_by_id(float_id: str) -> pd.DataFrame:
    loader = argopy.DataFetcher(src="erddap", mode="standard").float(int(float_id))
    ds = loader.to_xarray()
    df = ds.to_dataframe()
    return _clean(df.reset_index())