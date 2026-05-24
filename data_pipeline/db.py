# data_pipeline/db.py — Supabase PostgreSQL cache
# DB is optional for local dev — app starts without it, uses DB when available

import psycopg2
import psycopg2.extras
import pandas as pd
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
    """Creates cache table if it doesn't exist. Skips silently if DB unreachable."""
    if not is_db_available():
        print("DB unavailable — skipping table setup. Running without cache.")
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS argo_profiles (
            id SERIAL PRIMARY KEY,
            float_id VARCHAR,
            lat FLOAT,
            lon FLOAT,
            date DATE,
            pressure_dbar FLOAT,
            temperature_c FLOAT,
            salinity_psu FLOAT
        );
        CREATE INDEX IF NOT EXISTS idx_float_id ON argo_profiles(float_id);
        CREATE INDEX IF NOT EXISTS idx_date ON argo_profiles(date);
    """)
    conn.commit()
    cur.close()
    conn.close()


def cache_profiles(df: pd.DataFrame):
    """Inserts fetched profiles into cache. Skips silently if DB unreachable."""
    if not is_db_available():
        return
    conn = get_conn()
    cur = conn.cursor()
    rows = df.to_dict(orient="records")
    psycopg2.extras.execute_values(cur, """
        INSERT INTO argo_profiles
            (float_id, lat, lon, date, pressure_dbar, temperature_c, salinity_psu)
        VALUES %s ON CONFLICT DO NOTHING
    """, [(r["float_id"], r["lat"], r["lon"], r["date"],
           r["pressure_dbar"], r["temperature_c"], r["salinity_psu"]) for r in rows])
    conn.commit()
    cur.close()
    conn.close()


def query_profiles(sql: str) -> list[dict]:
    """Executes a SELECT against cached profiles. Returns empty list if DB unreachable."""
    if not is_db_available():
        return []
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows