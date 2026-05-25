# data_pipeline/db.py — Supabase PostgreSQL source of truth
# every fetch_region / fetch_float call writes here permanently
# over time this becomes the primary data source, reducing ERDDAP calls

import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import date
from config import DATABASE_URL


def get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def is_db_available() -> bool:
    """Non-fatal connectivity check — called before any DB operation."""
    try:
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False


def setup_tables():
    """
    Creates the profiles table and all indexes if they don't exist.
    Safe to call on every startup — all statements are idempotent.
    Skips silently if DB is unreachable.

    Also applies the UNIQUE constraint migration if the table already existed
    without it — uses a DO $$ block so it's safe to run multiple times.
    """
    if not is_db_available():
        print("DB unavailable — skipping table setup. Running without cache.")
        return

    conn = get_conn()
    cur = conn.cursor()

    # ── create tables ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS argo_profiles (
            id              SERIAL PRIMARY KEY,
            float_id        VARCHAR       NOT NULL,
            lat             FLOAT         NOT NULL,
            lon             FLOAT         NOT NULL,
            date            DATE          NOT NULL,
            pressure_dbar   FLOAT         NOT NULL,
            temperature_c   FLOAT,
            salinity_psu    FLOAT
        );

        -- spatial + temporal indexes for fast filtered queries
        CREATE INDEX IF NOT EXISTS idx_lat        ON argo_profiles(lat);
        CREATE INDEX IF NOT EXISTS idx_lon        ON argo_profiles(lon);
        CREATE INDEX IF NOT EXISTS idx_date       ON argo_profiles(date);
        CREATE INDEX IF NOT EXISTS idx_float_id   ON argo_profiles(float_id);
        -- composite index for the most common query pattern: region + date range
        CREATE INDEX IF NOT EXISTS idx_region_date
            ON argo_profiles(lat, lon, date);

        -- tracks which regions have already been fetched from ERDDAP
        -- used by fetcher.py to skip re-fetching data we already have
        CREATE TABLE IF NOT EXISTS fetched_regions (
            id          SERIAL PRIMARY KEY,
            lat_min     FLOAT   NOT NULL,
            lat_max     FLOAT   NOT NULL,
            lon_min     FLOAT   NOT NULL,
            lon_max     FLOAT   NOT NULL,
            date_start  DATE    NOT NULL,
            date_end    DATE    NOT NULL,
            fetched_at  TIMESTAMP DEFAULT NOW(),
            row_count   INT     DEFAULT 0
        );
    """)

    # ── UNIQUE constraint migration ────────────────────────────────────────────
    # The table may already exist without this constraint (deployed before the
    # migration was added). We add it conditionally so setup_tables() is always
    # safe to call — on a fresh DB the CREATE TABLE above already includes it
    # via the ALTER below; on an existing DB we add it now if it's missing.
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM   pg_constraint
                WHERE  conname = 'argo_profiles_float_id_date_pressure_dbar_key'
                AND    conrelid = 'argo_profiles'::regclass
            ) THEN
                ALTER TABLE argo_profiles
                    ADD CONSTRAINT argo_profiles_float_id_date_pressure_dbar_key
                    UNIQUE (float_id, date, pressure_dbar);
            END IF;
        END
        $$;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("DB tables, indexes, and constraints ready.")


def cache_profiles(df: pd.DataFrame):
    """
    Permanently stores fetched profiles.
    Uses ON CONFLICT DO NOTHING so re-fetching the same region is safe.
    Skips silently if DB is unreachable.
    """
    if not is_db_available() or df.empty:
        return

    conn = get_conn()
    cur = conn.cursor()

    rows = df.to_dict(orient="records")
    psycopg2.extras.execute_values(cur, """
        INSERT INTO argo_profiles
            (float_id, lat, lon, date, pressure_dbar, temperature_c, salinity_psu)
        VALUES %s
        ON CONFLICT (float_id, date, pressure_dbar) DO NOTHING
    """, [
        (r["float_id"], r["lat"], r["lon"], r["date"],
         r["pressure_dbar"], r["temperature_c"], r["salinity_psu"])
        for r in rows
    ])

    conn.commit()
    cur.close()
    conn.close()
    print(f"Stored {len(rows)} profile rows to DB.")


def log_fetched_region(lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        date_start: str, date_end: str,
                        row_count: int):
    """
    Records that a region+date range has been fetched from ERDDAP.
    Used by is_region_cached() to avoid duplicate ERDDAP calls.
    """
    if not is_db_available():
        return
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO fetched_regions
                (lat_min, lat_max, lon_min, lon_max, date_start, date_end, row_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (lat_min, lat_max, lon_min, lon_max, date_start, date_end, row_count))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"log_fetched_region failed (non-fatal): {e}")


def is_region_cached(lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float,
                      date_start: str, date_end: str) -> bool:
    """
    Returns True if this exact region+date range was previously fetched.
    Prevents redundant ERDDAP calls for repeated queries.
    Exact match only — a slightly different bounding box will still hit ERDDAP.
    """
    if not is_db_available():
        return False
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT 1 FROM fetched_regions
            WHERE lat_min = %s AND lat_max = %s
              AND lon_min = %s AND lon_max = %s
              AND date_start = %s AND date_end = %s
            LIMIT 1
        """, (lat_min, lat_max, lon_min, lon_max, date_start, date_end))
        found = cur.fetchone() is not None
        cur.close()
        conn.close()
        return found
    except Exception:
        return False


def load_profiles_from_db(lat_min: float, lat_max: float,
                           lon_min: float, lon_max: float,
                           date_start: str, date_end: str) -> pd.DataFrame:
    """
    Loads profiles for a region+date range directly from DB.
    Called by fetcher.py when is_region_cached() returns True.
    Returns empty DataFrame if DB is unavailable.
    """
    if not is_db_available():
        return pd.DataFrame()
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT float_id, lat, lon, date,
                   pressure_dbar, temperature_c, salinity_psu
            FROM argo_profiles
            WHERE lat  BETWEEN %s AND %s
              AND lon  BETWEEN %s AND %s
              AND date BETWEEN %s AND %s
            ORDER BY date, float_id, pressure_dbar
        """, (lat_min, lat_max, lon_min, lon_max, date_start, date_end))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["float_id"] = df["float_id"].astype(str)
        return df
    except Exception as e:
        print(f"load_profiles_from_db failed: {e}")
        return pd.DataFrame()


def load_float_from_db(float_id: str) -> pd.DataFrame:
    """
    Loads full profile history for a specific float from DB.
    Returns empty DataFrame if not found or DB unavailable.
    """
    if not is_db_available():
        return pd.DataFrame()
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT float_id, lat, lon, date,
                   pressure_dbar, temperature_c, salinity_psu
            FROM argo_profiles
            WHERE float_id = %s
            ORDER BY date, pressure_dbar
        """, (float_id,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["float_id"] = df["float_id"].astype(str)
        return df
    except Exception as e:
        print(f"load_float_from_db failed: {e}")
        return pd.DataFrame()


def query_profiles(sql: str) -> list[dict]:
    """
    Executes a SELECT against stored profiles.
    Called by the db_query tool in agents/tools.py.
    Returns empty list if DB unreachable.
    Only SELECT allowed — no mutations.
    """
    if not is_db_available():
        return []
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    # normalise date objects to strings for JSON serialisation
    for r in rows:
        if "date" in r and isinstance(r["date"], date):
            r["date"] = str(r["date"])
    return rows