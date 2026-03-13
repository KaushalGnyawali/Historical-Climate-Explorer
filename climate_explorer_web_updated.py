#!/usr/bin/env python3
"""
climate_explorer_web.py — Multi-Source Climate Data Explorer
=============================================================
Downloads daily climate data from ERA5-Land (9 km), NASA POWER (50 km),
GPM IMERG (10 km), and ECCC weather stations. Generates interactive Plotly
dashboards with three-season hydrological classification.

  python climate_explorer_web.py
  python climate_explorer_web.py --project SiteName --lat 49.5 --lon -121.3 \\
      --start 2000 --end 2025 --no-imerg

Requires: requests pandas numpy plotly
"""

import argparse, sys, time, os, getpass, json, math
from pathlib import Path
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════
DEFAULT_LAT, DEFAULT_LON = 50.0605, -122.9597   # Whistler Mountain, BC
OM_DELAY, OM_RETRIES, OM_BACKOFF = 3.5, 4, 20.0
NP_DELAY = 1.5
IMERG_WORKERS = 8
IMERG_START_DATE = date(2000, 6, 1)
# Final Run: gauge-corrected, ~3.5-month production lag
IMERG_BASE_FINAL = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.07"
# Late Run: satellite-only, ~12-hour latency — fills the recent gap
IMERG_BASE_LATE  = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.07"
IMERG_FINAL_LATENCY_DAYS = 107  # ~3.5 months — switch to Late Run beyond this

SEASONS = {
    "Snow":     {"months": [11, 12, 1, 2, 3], "color": "#3498db", "emoji": "❄️"},
    "Melt":     {"months": [4, 5],             "color": "#f39c12", "emoji": "💧"},
    "Rainfall": {"months": [6, 7, 8, 9, 10],  "color": "#e74c3c", "emoji": "☀️"},
}
SEASON_ORDER = ["Snow", "Melt", "Rainfall"]

# ═════════════════════════════════════════════════════════════════
# VARIABLE REGISTRY  (every plottable column)
# ═════════════════════════════════════════════════════════════════
VMETA = {
    # ── ERA5-Land ──
    "precip_era5":     {"name": "Precipitation (ERA5-Land)", "unit": "mm/day",
                        "color": "#2471a3", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best gridded — 9 km orographic enhancement"},
    "rain":            {"name": "Rainfall", "unit": "mm/day",
                        "color": "#27ae60", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "Liquid-phase only"},
    "snowfall":        {"name": "Snowfall", "unit": "cm/day",
                        "color": "#aeb6bf", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "Solid-phase"},
    "temp_mean":       {"name": "Mean Temperature", "unit": "°C",
                        "color": "#e74c3c", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best — 9 km lapse rates"},
    "temp_max":        {"name": "Max Temperature", "unit": "°C",
                        "color": "#c0392b", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best — diurnal extremes"},
    "temp_min":        {"name": "Min Temperature", "unit": "°C",
                        "color": "#2980b9", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best — cold-air pooling"},
    "wind_max":        {"name": "Max Wind Speed", "unit": "m/s",
                        "color": "#e67e22", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best — valley channelling"},
    "wind_gust":       {"name": "Max Wind Gust", "unit": "m/s",
                        "color": "#d35400", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "Gust parameterisation"},
    "solar_rad":       {"name": "Solar Radiation", "unit": "MJ/m²",
                        "color": "#f1c40f", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Best — terrain shading"},
    "et0":             {"name": "Reference ET₀ (FAO)", "unit": "mm/day",
                        "color": "#f39c12", "src": "ERA5Land",
                        "src_full": "ERA5-Land (ECMWF, 9 km)",
                        "quality": "★ Penman-Monteith at 9 km"},
    # ── NASA POWER ──
    "precip_power":    {"name": "Precipitation (POWER)", "unit": "mm/day",
                        "color": "#1a5276", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "IMERG-corrected; 50 km smooths extremes"},
    "snow_depth":      {"name": "Snow Depth", "unit": "m",
                        "color": "#5dade2", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "★ Only gridded source"},
    "soil_moist_sfc":  {"name": "Soil Moisture (Surface)", "unit": "[0-1]",
                        "color": "#8e44ad", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "★ Only source — antecedent moisture"},
    "soil_moist_root": {"name": "Soil Moisture (Root Zone)", "unit": "[0-1]",
                        "color": "#6c3483", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "★ Only source — root-zone saturation"},
    "soil_moist_prof": {"name": "Soil Moisture (Profile)", "unit": "[0-1]",
                        "color": "#4a235a", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "★ Only source — deep drainage"},
    "humidity":        {"name": "Relative Humidity", "unit": "%",
                        "color": "#1abc9c", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "50 km adequate"},
    "dewpoint":        {"name": "Dewpoint", "unit": "°C",
                        "color": "#16a085", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "Fog/rime; 50 km adequate"},
    "pressure":        {"name": "Surface Pressure", "unit": "hPa",
                        "color": "#34495e", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "Synoptic; 50 km adequate"},
    "cloud_cover":     {"name": "Cloud Cover", "unit": "%",
                        "color": "#95a5a6", "src": "POWER",
                        "src_full": "NASA POWER (CERES, 50 km)",
                        "quality": "CERES satellite"},
    "wind_mean":       {"name": "Mean Wind Speed", "unit": "m/s",
                        "color": "#eb984e", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "50 km misses channelling"},
    "temp_range":      {"name": "Diurnal Temp Range", "unit": "°C",
                        "color": "#a93226", "src": "POWER",
                        "src_full": "NASA POWER (MERRA-2, 50 km)",
                        "quality": "Freeze-thaw proxy"},
    # ── GPM IMERG ──
    "precip_imerg":    {"name": "Precipitation (GPM IMERG)", "unit": "mm/day",
                        "color": "#154360", "src": "IMERG",
                        "src_full": "GPM IMERG V07 (satellite, 10 km)",
                        "quality": "★ Independent satellite validation"},
    # ── ECCC Stations ──
    "precip_eccc":     {"name": "Precipitation (ECCC Station)", "unit": "mm/day",
                        "color": "#117864", "src": "ECCC",
                        "src_full": "Environment Canada gauge (point)",
                        "quality": "★★ Ground truth — rain gauge"},
    "temp_mean_eccc":  {"name": "Mean Temp (ECCC)", "unit": "°C",
                        "color": "#b03a2e", "src": "ECCC",
                        "src_full": "Environment Canada gauge (point)",
                        "quality": "★★ Ground truth"},
    "temp_max_eccc":   {"name": "Max Temp (ECCC)", "unit": "°C",
                        "color": "#922b21", "src": "ECCC",
                        "src_full": "Environment Canada gauge (point)",
                        "quality": "★★ Ground truth"},
    "temp_min_eccc":   {"name": "Min Temp (ECCC)", "unit": "°C",
                        "color": "#1a5276", "src": "ECCC",
                        "src_full": "Environment Canada gauge (point)",
                        "quality": "★★ Ground truth"},
    "snow_ground_eccc": {"name": "Snow on Ground (ECCC)", "unit": "cm",
                         "color": "#85c1e9", "src": "ECCC",
                         "src_full": "Environment Canada gauge (point)",
                         "quality": "★★ Ground truth — manual depth"},
}

# Dashboard variable lists per source
DASH_ERA5 = ["precip_era5", "rain", "snowfall", "temp_mean", "temp_max",
             "temp_min", "wind_max", "wind_gust", "solar_rad", "et0"]
DASH_POWER = ["precip_power", "snow_depth", "soil_moist_sfc", "soil_moist_root",
              "soil_moist_prof", "humidity", "dewpoint", "pressure",
              "cloud_cover", "wind_mean", "temp_range"]
DASH_IMERG = ["precip_imerg"]
DASH_ECCC = ["precip_eccc", "temp_mean_eccc", "temp_max_eccc",
             "temp_min_eccc", "snow_ground_eccc"]
DASH_COMBINED = [
    "precip_era5", "precip_power", "precip_imerg", "precip_eccc",
    "temp_mean", "temp_max", "temp_min", "temp_mean_eccc",
    "snow_depth", "snow_ground_eccc", "wind_max", "wind_gust",
    "soil_moist_sfc", "soil_moist_root", "soil_moist_prof",
    "humidity", "dewpoint", "pressure", "cloud_cover", "solar_rad", "et0",
]

# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════
def loc_tag(lat, lon):
    return (f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}_"
            f"{abs(lon):.4f}{'W' if lon < 0 else 'E'}")

def loc_display(lat, lon):
    return (f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.4f}°{'W' if lon < 0 else 'E'}")

def safe_col(df, col):
    return col in df.columns and df[col].dropna().shape[0] > 10

def get_season(m):
    for sn, si in SEASONS.items():
        if m in si["months"]:
            return sn
    return "Rainfall"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

def retry_get(url, params, retries=3, backoff=15.0, timeout=60):
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                if i < retries:
                    d = backoff * (2 ** i)
                    print(f" ⏳{d:.0f}s…", end=" ", flush=True)
                    time.sleep(d)
                    continue
                return None
            if r.status_code >= 500 and i < retries:
                time.sleep(backoff * (2 ** i))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if i < retries:
                time.sleep(backoff * (2 ** i))
            else:
                print(f" FAIL: {e}")
                return None
    return None

def validate_df(df, name):
    """Print validation summary for a DataFrame."""
    print(f"\n  ── Validation: {name} ──")
    print(f"     Rows: {len(df):,}")
    if "date" in df.columns and len(df) > 0:
        print(f"     Dates: {df['date'].min()} → {df['date'].max()}")
    dcols = [c for c in df.columns if c != "date"]
    if dcols:
        hdr = f"     {'Column':<25s} {'Non-null':>8s} {'Null':>6s} {'Mean':>10s} {'Min':>10s} {'Max':>10s}"
        print(hdr)
        for c in dcols:
            nn = df[c].notna().sum()
            nu = df[c].isna().sum()
            if nn > 0:
                print(f"     {c:<25s} {nn:>8,d} {nu:>6,d} "
                      f"{df[c].mean():>10.3f} {df[c].min():>10.3f} {df[c].max():>10.3f}")
            else:
                print(f"     {c:<25s} {nn:>8,d} {nu:>6,d} "
                      f"{'—':>10s} {'—':>10s} {'—':>10s}")

# ═════════════════════════════════════════════════════════════════
# CACHING
# ═════════════════════════════════════════════════════════════════
def check_cached(raw_dir, prefix, lat, lon):
    fp = raw_dir / f"{prefix}_daily_{loc_tag(lat, lon)}.csv"
    if fp.exists() and fp.stat().st_size > 200:
        try:
            df = pd.read_csv(fp, nrows=20, parse_dates=["date"])
            dcols = [c for c in df.columns if c != "date"]
            if dcols and df[dcols].dropna(how="all").shape[0] == 0:
                print(f"  ⚠️ Cached {prefix} has no data — will re-download.")
                return None
            return fp
        except Exception:
            pass
    return None

def load_cached(fp):
    try:
        df = pd.read_csv(fp, parse_dates=["date"])
        dcols = [c for c in df.columns if c != "date"]
        valid = df[dcols].notna().any(axis=1).sum() if dcols else 0
        print(f"  📦 Loaded {fp.name} ({len(df):,} rows, {valid:,} with data)")
        return df
    except Exception as e:
        print(f"  ⚠️ Cache fail: {e}")
        return pd.DataFrame()

def ask_reuse(name, fp, default_yes=False):
    mod = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    kb = fp.stat().st_size / 1024
    yn = "Y/n" if default_yes else "y/N"
    try:
        s = input(f"  📦 Cached {name}: {fp.name} "
                  f"({kb:.0f}KB, {mod}). Reuse? [{yn}]: ").strip().lower()
        if not s:
            return default_yes
        return s in ("y", "yes")
    except Exception:
        return default_yes

# ═════════════════════════════════════════════════════════════════
# PROMPTS
# ═════════════════════════════════════════════════════════════════
def prompt_settings(args):
    cur = datetime.now().year
    lat, lon = args.lat, args.lon

    print(f"\n{'=' * 60}")
    print(f"  Climate Data Explorer")
    print(f"{'=' * 60}")
    print(f"  ERA5-Land 9 km | POWER 50 km | IMERG 10 km | ECCC\n")

    # Project folder
    project = args.project
    if not project and not args.no_prompts:
        try:
            project = (input("  📂 Project folder [Whistler_Climate]: ").strip()
                       or "Whistler_Climate")
        except Exception:
            project = "Whistler_Climate"
    if not project:
        project = "Whistler_Climate"

    # Coordinates
    if not args.lat_set:
        try:
            s = input(f"  📍 Latitude  [{DEFAULT_LAT}]: ").strip()
            if s:
                lat = float(s)
            s = input(f"  📍 Longitude [{DEFAULT_LON}]: ").strip()
            if s:
                lon = float(s)
        except Exception:
            pass

    # Date range
    start_yr, end_yr = args.start, args.end
    if not args.start_set:
        try:
            s = input(f"  📅 Start year [2000]: ").strip()
            if s:
                start_yr = int(s)
            s = input(f"  📅 End year   [{cur}]: ").strip()
            if s:
                end_yr = int(s)
        except Exception:
            pass
    end_yr = min(end_yr, cur)
    if start_yr < 1950:
        start_yr = 1950

    # IMERG
    do_imerg = not args.no_imerg
    imerg_start = args.imerg_start
    imerg_end = args.imerg_end
    imerg_latest = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    def _parse_imerg_date(s, fallback_suffix):
        """Accept YYYY → YYYY-{fallback_suffix}, or full YYYY-MM-DD as-is."""
        s = s.strip()
        if not s:
            return None
        if len(s) == 4 and s.isdigit():
            return f"{s}-{fallback_suffix}"
        return s  # already a full date string

    if do_imerg and not args.no_imerg_prompt:
        print(f"\n  🛰️ GPM IMERG — satellite precipitation (10 km)")
        print(f"     Requires free NASA Earthdata account. ~1 min/yr.")
        try:
            s = input("     Include IMERG? [y/N]: ").strip().lower()
            if s in ("y", "yes"):
                im_def = max(start_yr, 2000)
                s = input(f"     IMERG start [{im_def}]: ").strip()
                imerg_start = _parse_imerg_date(s, "01-01") or f"{im_def}-06-01"
                s = input(f"     IMERG end   [{imerg_latest}]: ").strip()
                imerg_end = _parse_imerg_date(s, "12-31") or imerg_latest
            else:
                do_imerg = False
        except Exception as e:
            print(f"  ⚠️ IMERG prompt error: {e} — skipping IMERG")
            do_imerg = False
    elif do_imerg and args.no_imerg_prompt:
        # Creds supplied or --no-prompts: auto-fill missing dates
        if not imerg_start:
            im_def = max(start_yr, 2000)
            imerg_start = f"{im_def}-06-01"
        if not imerg_end:
            imerg_end = imerg_latest
        print(f"  🛰️ IMERG auto: {imerg_start} → {imerg_end}")

    # ECCC
    do_eccc = not args.no_eccc
    if do_eccc and not args.no_eccc_prompt:
        try:
            s = input("\n  🍁 Include Environment Canada stations? "
                      "(free, fast) [Y/n]: ").strip().lower()
            if s in ("n", "no"):
                do_eccc = False
        except Exception:
            pass

    print(f"\n  ✅ {loc_display(lat, lon)} | {start_yr}→{end_yr} | {project}/")
    print(f"     IMERG: {'yes' if do_imerg else 'no'} | "
          f"ECCC: {'yes' if do_eccc else 'no'}")
    return (lat, lon, start_yr, end_yr, project,
            do_imerg, imerg_start, imerg_end, do_eccc)

# ═════════════════════════════════════════════════════════════════
# SOURCE 1 — ERA5-LAND  (Open-Meteo archive API, 9 km)
# ═════════════════════════════════════════════════════════════════
def download_era5land(lat, lon, start_yr, end_yr):
    print(f"\n{'─' * 70}")
    print(f"  📡 ERA5-Land (9 km) {start_yr}–{end_yr}")
    print(f"{'─' * 70}")
    apivars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "rain_sum", "snowfall_sum",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    ]
    minimal = ("temperature_2m_max,temperature_2m_min,"
               "precipitation_sum,wind_speed_10m_max")
    max_date = (date.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    max_yr = int(max_date[:4])
    print(f"  ℹ️ Archive through ~{max_date}")
    frames = []
    for cs in range(start_yr, end_yr + 1, 5):
        ce = min(cs + 4, end_yr)
        chunk_end = max_date if ce >= max_yr else f"{ce}-12-31"
        if cs > max_yr:
            continue
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{cs}-01-01", "end_date": chunk_end,
            "daily": ",".join(apivars), "timezone": "auto",
        }
        print(f"  ⬇️ {cs}→{min(ce, max_yr)}…", end=" ", flush=True)
        r = retry_get(
            "https://archive-api.open-meteo.com/v1/archive",
            params, retries=OM_RETRIES, backoff=OM_BACKOFF,
        )
        if r is None:
            params["daily"] = minimal
            time.sleep(OM_BACKOFF)
            print("↻min…", end=" ", flush=True)
            r = retry_get(
                "https://archive-api.open-meteo.com/v1/archive",
                params, retries=2, backoff=OM_BACKOFF * 2,
            )
        if r is None:
            continue
        try:
            d = r.json()
            if "daily" not in d:
                print(f"⚠️ {d.get('reason', '')}")
                continue
            c = pd.DataFrame(d["daily"])
            c["time"] = pd.to_datetime(c["time"])
            frames.append(c)
            print(f"✅ ({len(c)})")
        except Exception as e:
            print(f"⚠️ {e}")
        time.sleep(OM_DELAY)

    if not frames:
        print("  ❌ No ERA5-Land data.")
        return pd.DataFrame()
    df = (pd.concat(frames, ignore_index=True)
          .sort_values("time").drop_duplicates("time")
          .reset_index(drop=True))
    rmap = {
        "time": "date",
        "temperature_2m_mean": "temp_mean",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip_era5",
        "rain_sum": "rain",
        "snowfall_sum": "snowfall",
        "wind_speed_10m_max": "wind_max",
        "wind_gusts_10m_max": "wind_gust",
        "shortwave_radiation_sum": "solar_rad",
        "et0_fao_evapotranspiration": "et0",
    }
    df = df.rename(columns={k: v for k, v in rmap.items() if k in df.columns})
    if "temp_mean" not in df.columns and "temp_max" in df.columns:
        df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
    validate_df(df, "ERA5-Land")
    return df

# ═════════════════════════════════════════════════════════════════
# SOURCE 2 — NASA POWER  (MERRA-2 / GEOS / CERES, 50 km)
# ═════════════════════════════════════════════════════════════════
def download_power(lat, lon, start_yr, end_yr):
    print(f"\n{'─' * 70}")
    print(f"  📡 NASA POWER (50 km) {max(start_yr, 1981)}–{end_yr}")
    print(f"{'─' * 70}")
    full = [
        "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR",
        "WS10M", "WS10M_MAX", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT",
        "PS", "GWETTOP", "GWETROOT", "GWETPROF", "SNODP", "T2M_RANGE",
    ]
    reduced = [
        "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "WS10M_MAX",
        "RH2M", "SNODP", "GWETTOP", "PS",
    ]
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    frames = []
    today = date.today()
    for cs in range(max(start_yr, 1981), end_yr + 1, 10):
        ce = min(cs + 9, end_yr)
        # Cap end at today — POWER will reject future dates
        chunk_end = date(ce, 12, 31)
        if chunk_end > today:
            chunk_end = today
        params = {
            "parameters": ",".join(full), "community": "RE",
            "longitude": lon, "latitude": lat,
            "start": f"{cs}0101",
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        print(f"  ⬇️ {cs}–{ce}…", end=" ", flush=True)
        for att, pl in enumerate([full, reduced]):
            try:
                params["parameters"] = ",".join(pl)
                if att:
                    print("↻reduced…", end=" ", flush=True)
                    time.sleep(5)
                r = requests.get(base, params=params, timeout=180)
                r.raise_for_status()
                d = r.json()
                if "properties" not in d:
                    print("⚠️ bad")
                    break
                recs = {}
                for pn, dv in d["properties"]["parameter"].items():
                    for ds, val in dv.items():
                        if ds not in recs:
                            recs[ds] = {}
                        recs[ds][pn] = val if val != -999.0 else np.nan
                c = pd.DataFrame.from_dict(recs, orient="index")
                c.index = pd.to_datetime(c.index, format="%Y%m%d")
                c = (c.sort_index().reset_index()
                     .rename(columns={"index": "date"}))
                frames.append(c)
                print(f"✅ ({len(c)})")
                break
            except Exception as e:
                if att == 0:
                    print(f"⚠️ {e}")
                else:
                    print(f"FAIL: {e}")
        time.sleep(NP_DELAY)

    if not frames:
        print("  ❌ No POWER data.")
        return pd.DataFrame()
    df = (pd.concat(frames, ignore_index=True)
          .sort_values("date").drop_duplicates("date")
          .reset_index(drop=True))
    rmap = {
        "PRECTOTCORR": "precip_power", "T2M": "_pw_temp_mean",
        "T2M_MAX": "_pw_temp_max", "T2M_MIN": "_pw_temp_min",
        "T2MDEW": "dewpoint", "RH2M": "humidity",
        "WS10M": "wind_mean", "WS10M_MAX": "_pw_wind_max",
        "ALLSKY_SFC_SW_DWN": "_pw_solar_rad", "CLOUD_AMT": "cloud_cover",
        "PS": "pressure", "GWETTOP": "soil_moist_sfc",
        "GWETROOT": "soil_moist_root", "GWETPROF": "soil_moist_prof",
        "SNODP": "snow_depth", "T2M_RANGE": "temp_range",
    }
    df = df.rename(columns={k: v for k, v in rmap.items() if k in df.columns})
    if "pressure" in df.columns:
        df["pressure"] *= 10  # kPa → hPa
    validate_df(df, "NASA POWER")
    return df

# ═════════════════════════════════════════════════════════════════
# SOURCE 3 — GPM IMERG  (satellite, 10 km, OPeNDAP)
# ═════════════════════════════════════════════════════════════════

class EarthdataSession(requests.Session):
    """
    Override rebuild_auth so credentials survive the NASA URS redirect.

    Python requests strips the Authorization header on cross-host redirects
    (security feature).  NASA GES DISC redirects to urs.earthdata.nasa.gov
    for OAuth — the header gets stripped — URS never sees credentials —
    returns HTML login page.  This override keeps auth for URS redirects.
    """
    def rebuild_auth(self, prepared_request, response):
        if "urs.earthdata.nasa.gov" in (prepared_request.url or ""):
            if self.auth:
                prepared_request.prepare_auth(self.auth)
            return
        super().rebuild_auth(prepared_request, response)


def get_earthdata_creds(cu="", cp=""):
    u = cu or os.environ.get("EARTHDATA_USERNAME", "")
    p = cp or os.environ.get("EARTHDATA_PASSWORD", "")
    if u and p:
        print(f"  🔑 Earthdata creds OK.")
        return u, p
    print()
    print("  🛰️ NASA Earthdata login for IMERG")
    print("     Signup:  https://urs.earthdata.nasa.gov/users/new")
    print("     APPROVE: https://disc.gsfc.nasa.gov/earthdata-login")
    print()
    try:
        u = input("  Username: ").strip()
        if not u:
            print("  ⏭️ Skip.")
            return None, None
        p = getpass.getpass("  Password: ").strip()
        if not p:
            print("  ⏭️ Skip.")
            return None, None
    except Exception:
        print("\n  ⏭️ Skip.")
        return None, None
    return u, p


def _parse_imerg_ascii(text):
    """
    Parse OPeNDAP ASCII response for IMERG precipitation.

    Response format (Grid sub-array):
      Dataset: 3B-DAY.MS.MRG.3IMERG.20200701-S000000-E235959.V07B.nc4
      precipitation.precipitation[0][0], 21.325

    Response format (full Grid):
      Dataset: ...
      precipitation.lat, 56.15
      precipitation.precipitation[precipitation.time=14787][precipitation.lon=-120.75], 21.325
    """
    for line in text.strip().split("\n"):
        s = line.strip()
        # Skip empty lines and Dataset header
        if not s or s.startswith("Dataset"):
            continue
        # Look for lines with a comma that contain "precipitation"
        # Both formats have: ..., VALUE  at the end
        if "precipitation" in s.lower() and "," in s:
            # Skip the .lat and .lon map lines
            if ".lat," in s or ".lon," in s or ".time," in s:
                continue
            # Extract value after the LAST comma
            try:
                val = float(s.rsplit(",", 1)[-1].strip())
                if val >= 0:
                    return val
            except ValueError:
                continue
    return np.nan


def test_imerg_auth(username, password):
    """Hit a known IMERG file to verify auth + data access."""
    print("  🔐 Testing IMERG auth…", end=" ", flush=True)
    session = EarthdataSession()
    session.auth = (username, password)
    session.headers.update({"User-Agent": "climate_explorer/4.5"})

    #  ┌──────────────────────────────────────────────────────────────┐
    #  │  DDS: precipitation[time=1][lon=3600][lat=1800]  (Grid)     │
    #  │  Query order: [time][lon][lat]                              │
    #  │  Use dot syntax: precipitation.precipitation to get only    │
    #  │  the data array without map vectors.                        │
    #  └──────────────────────────────────────────────────────────────┘
    test_url = (
        f"{IMERG_BASE_FINAL}/2020/07/"
        f"3B-DAY.MS.MRG.3IMERG.20200701-S000000-E235959.V07B.nc4"
        f".ascii?precipitation.precipitation[0][1800][900]"
    )
    try:
        r = session.get(test_url, timeout=60, allow_redirects=True)
        if r.status_code == 401:
            print("❌ 401 — wrong password.")
            return False
        if r.status_code == 403:
            print("❌ 403 — approve GES DISC:")
            print("     https://disc.gsfc.nasa.gov/earthdata-login")
            return False
        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code}")
            return False
        text = r.text
        if "<html" in text.lower()[:500]:
            print("❌ Got HTML login page. Approve GES DISC:")
            print("     https://disc.gsfc.nasa.gov/earthdata-login")
            return False
        val = _parse_imerg_ascii(text)
        if not np.isnan(val):
            print(f"✅ Test = {val:.2f} mm/day")
            return True
        else:
            print("⚠️ Parsed NaN — response:")
            for ln in text[:500].split("\n")[:10]:
                print(f"       | {ln.rstrip()}")
            return True  # might be a dry day (value = 0)
    except Exception as e:
        print(f"⚠️ {e}")
        return True  # don't block on network hiccup


IMERG_DAY_RETRIES = 3
IMERG_DAY_BACKOFF = 5.0

def _fetch_imerg_day(args_tuple):
    """
    Fetch one day of IMERG precipitation via OPeNDAP with retries.

    DDS structure:  precipitation[time=1][lon=3600][lat=1800]  (Grid)
    Query order:    precipitation.precipitation[0][lon_idx][lat_idx]
    Uses dot syntax to get only the data array (no map vectors).

    Supports both Final Run (gauge-corrected, ~3.5 month lag) and
    Late Run (satellite-only, ~12 hour lag) products.
    """
    d, lat_idx, lon_idx, auth, debug_dir, imerg_base = args_tuple
    yyyy = d.strftime("%Y")
    mm = d.strftime("%m")
    yyyymmdd = d.strftime("%Y%m%d")

    # Filename differs between Final Run and Late Run
    if "IMERGDL" in imerg_base:
        filename = f"3B-DAY-L.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"
    else:
        filename = f"3B-DAY.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"

    # NOTE: dimension order is [time][LON][LAT] per DDS
    url = (
        f"{imerg_base}/{yyyy}/{mm}/{filename}"
        f".ascii?precipitation.precipitation[0][{lon_idx}][{lat_idx}]"
    )

    val = np.nan
    for attempt in range(IMERG_DAY_RETRIES):
        try:
            s = EarthdataSession()
            s.auth = auth
            s.headers.update({"User-Agent": "climate_explorer/4.5"})
            r = s.get(url, timeout=90, allow_redirects=True)

            if r.status_code == 200:
                text = r.text
                # save debug for first few requests
                if debug_dir:
                    try:
                        (Path(debug_dir) / f"imerg_{yyyymmdd}.txt").write_text(
                            f"URL: {url}\nHTTP: {r.status_code}\n"
                            f"CT: {r.headers.get('Content-Type', '')}\n\n"
                            f"{text[:2000]}"
                        )
                    except Exception:
                        pass
                if "<html" in text.lower()[:200]:
                    # Got login redirect — no point retrying with same creds
                    return {"date": pd.Timestamp(d), "precip_imerg": np.nan}
                raw = _parse_imerg_ascii(text)
                if not np.isnan(raw) and raw >= 0:
                    val = raw  # Already mm/day for daily product
                    break
            elif r.status_code in (429, 500, 502, 503, 504):
                # Retryable server errors
                if attempt < IMERG_DAY_RETRIES - 1:
                    time.sleep(IMERG_DAY_BACKOFF * (attempt + 1))
                    continue
            else:
                break  # Non-retryable HTTP error (401, 403, 404)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            if attempt < IMERG_DAY_RETRIES - 1:
                time.sleep(IMERG_DAY_BACKOFF * (attempt + 1))
                continue
        except Exception:
            break
    return {"date": pd.Timestamp(d), "precip_imerg": val}


def download_imerg(lat, lon, start_yr, end_yr, raw_dir,
                   cu="", cp="", ims=None, ime=None):
    """Download GPM IMERG V07 daily precipitation via OPeNDAP."""
    print(f"\n{'─' * 70}")
    print(f"  🛰️ GPM IMERG V07 (10 km) ⚡{IMERG_WORKERS} workers")
    print(f"{'─' * 70}")
    print(f"  ℹ️ IMERG = satellite precipitation only (no temp/wind/soil)\n")

    username, password = get_earthdata_creds(cu, cp)
    if not username:
        return pd.DataFrame()
    if not test_imerg_auth(username, password):
        print("\n  ❌ IMERG auth failed.")
        print("     Fix: https://disc.gsfc.nasa.gov/earthdata-login")
        return pd.DataFrame()

    auth = (username, password)

    # Grid indices  (IMERG: 0.1° from -89.95 to 89.95 / -179.95 to 179.95)
        # IMERG grid: lat -89.95→89.95 (1800), lon -179.95→179.95 (3600)
    # DDS dimension order: precipitation[time][LON][LAT]
    lat_idx = max(0, min(1799, int(round((lat - (-89.95)) / 0.1))))
    lon_idx = max(0, min(3599, int(round((lon - (-179.95)) / 0.1))))
    grid_lat = -89.95 + lat_idx * 0.1
    grid_lon = -179.95 + lon_idx * 0.1
    print(f"  🎯 Grid: ({grid_lat:.2f}°, {grid_lon:.2f}°)  "
          f"idx: lat={lat_idx}, lon={lon_idx}")
    print(f"  📐 Query order: precipitation.precipitation"
          f"[0][{lon_idx}][{lat_idx}]  (time/lon/lat)")    
    lat_idx = max(0, min(1799, int(round((lat - (-89.95)) / 0.1))))
    lon_idx = max(0, min(3599, int(round((lon - (-179.95)) / 0.1))))


    # ── Date range ──
    # Final Run (~3.5 month lag): gauge-corrected, use for all historical dates
    # Late Run  (~12 hour lag):   satellite-only, use for recent period
    # Cap at yesterday — Late Run needs ~12 h to process the previous day
    late_max = date.today() - timedelta(days=1)
    final_cutoff = date.today() - timedelta(days=IMERG_FINAL_LATENCY_DAYS)

    ds = (pd.to_datetime(ims).date() if ims
          else max(date(start_yr, 1, 1), IMERG_START_DATE))
    de = min(
        pd.to_datetime(ime).date() if ime else date(end_yr, 12, 31),
        late_max,
    )
    if ds < IMERG_START_DATE:
        ds = IMERG_START_DATE
    if ds > de:
        print("  ⚠️ No date range.")
        return pd.DataFrame()

    n_final = max(0, (min(de, final_cutoff) - ds).days + 1)
    n_late  = max(0, (de - max(ds, final_cutoff + timedelta(days=1))).days + 1)
    print(f"  📅 {ds} → {de}  (total {(de - ds).days + 1:,} days)")
    print(f"     Final Run (gauge-corrected): {ds} → {min(de, final_cutoff)}"
          f"  ({n_final:,} days)")
    if n_late > 0:
        print(f"     Late Run  (satellite-only):  "
              f"{max(ds, final_cutoff + timedelta(days=1))} → {de}"
              f"  ({n_late:,} days)  ⚠️ lower quality")

    total = (de - ds).days + 1

    # Resume cache
    cache_file = raw_dir / f".imerg_cache_{loc_tag(lat, lon)}.csv"
    cached_dates, cached_records = set(), []
    if cache_file.exists():
        try:
            cdf = pd.read_csv(cache_file, parse_dates=["date"])
            mask = ((cdf["date"].dt.date >= ds) &
                    (cdf["date"].dt.date <= de))
            cdf = cdf[mask]
            valid = cdf[cdf["precip_imerg"].notna()]
            if len(valid) > 0:
                cached_dates = set(valid["date"].dt.date)
                cached_records = valid.to_dict("records")
                print(f"  📦 Resume: {len(cached_dates):,} cached")
            else:
                print("  ⚠️ Old cache all NaN — fresh download")
        except Exception:
            pass

    needed = [ds + timedelta(days=i) for i in range(total)
              if (ds + timedelta(days=i)) not in cached_dates]

    if not needed:
        df = pd.DataFrame(cached_records)
        df["date"] = pd.to_datetime(df["date"])
        print(f"  ✅ All {total:,} days cached!")
        validate_df(df, "IMERG (cached)")
        return df.sort_values("date").reset_index(drop=True)

    print(f"  ⬇️ {len(needed):,} new days to fetch "
          f"(~{len(needed) / IMERG_WORKERS / 60:.1f} min)\n")

    # Debug dir
    debug_dir = raw_dir / "_imerg_debug"
    debug_dir.mkdir(exist_ok=True)
    dbg_left = [5]  # save first 5 raw responses

    tasks = []
    for d in needed:
        dbg = str(debug_dir) if dbg_left[0] > 0 else None
        if dbg:
            dbg_left[0] -= 1
        # Route to correct product based on date
        imerg_base = (IMERG_BASE_LATE if d > final_cutoff
                      else IMERG_BASE_FINAL)
        tasks.append((d, lat_idx, lon_idx, auth, dbg, imerg_base))

    new_records = []
    errors, valid_count = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=IMERG_WORKERS) as executor:
        futs = {executor.submit(_fetch_imerg_day, t): t[0] for t in tasks}
        done = 0

        for f in as_completed(futs):
            done += 1
            try:
                res = f.result()
                new_records.append(res)
                if not np.isnan(res.get("precip_imerg", np.nan)):
                    valid_count += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
                new_records.append({
                    "date": pd.Timestamp(futs[f]),
                    "precip_imerg": np.nan,
                })

            # Progress
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / el if el else 0
                rem = (len(tasks) - done) / rate / 60 if rate else 0
                pct = (done + len(cached_dates)) / total * 100
                print(f"    {pct:5.1f}% | "
                      f"{done + len(cached_dates):,}/{total:,} | "
                      f"✅{valid_count:,} ❌{errors} | "
                      f"~{rem:.1f}min")

            # Early abort
            if done == 50 and valid_count == 0:
                print(f"\n  ❌ ABORT: First 50 all failed!")
                print(f"     Check: {debug_dir}/")
                for ff in futs:
                    ff.cancel()
                break

            # Incremental save
            if done % 500 == 0:
                try:
                    pd.DataFrame(
                        cached_records + new_records
                    ).to_csv(cache_file, index=False)
                except Exception:
                    pass

    # Combine
    df = pd.DataFrame(cached_records + new_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # ── Gap-fill retry pass ──
    # Re-attempt NaN days — pass correct product URL for each date
    nan_dates = df[df["precip_imerg"].isna()]["date"].dt.date.tolist()
    if nan_dates and len(nan_dates) < len(df) * 0.5:
        # Only retry if less than half are NaN (otherwise it's likely auth issue)
        print(f"\n  🔄 Gap-fill: retrying {len(nan_dates):,} missing days…")
        retry_tasks = []
        for d in nan_dates:
            imerg_base = (IMERG_BASE_LATE if d > final_cutoff
                          else IMERG_BASE_FINAL)
            retry_tasks.append((d, lat_idx, lon_idx, auth, None, imerg_base))

        retry_records = []
        filled = 0
        with ThreadPoolExecutor(max_workers=min(4, IMERG_WORKERS)) as executor:
            futs = {executor.submit(_fetch_imerg_day, t): t[0]
                    for t in retry_tasks}
            for f in as_completed(futs):
                try:
                    res = f.result()
                    if not np.isnan(res.get("precip_imerg", np.nan)):
                        retry_records.append(res)
                        filled += 1
                except Exception:
                    pass

        if retry_records:
            retry_df = pd.DataFrame(retry_records)
            retry_df["date"] = pd.to_datetime(retry_df["date"])
            # Update NaN rows in df with retry results
            df = df.set_index("date")
            retry_df = retry_df.set_index("date")
            df.update(retry_df)
            df = df.reset_index()
            print(f"     ✅ Recovered {filled:,} days")
        else:
            print(f"     ⚠️ No recoveries")

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    try:
        df.to_csv(cache_file, index=False)
    except Exception:
        pass

    vt = df["precip_imerg"].dropna().shape[0]
    elapsed = (time.time() - t0) / 60

    if vt == 0:
        print(f"\n  ❌ ALL IMERG NaN!  Check {debug_dir}/")
        for fp in list(debug_dir.glob("*.txt"))[:3]:
            print(f"     → {fp}")
            try:
                print(f"       {fp.read_text()[:200]}")
            except Exception:
                pass
        print("     Fix: https://disc.gsfc.nasa.gov/earthdata-login")
    else:
        print(f"\n  ✅ IMERG: {len(df):,} days, "
              f"{vt:,} valid ({elapsed:.1f} min)")

    validate_df(df, "GPM IMERG")
    return df

# ═════════════════════════════════════════════════════════════════
# SOURCE 4 — ENVIRONMENT CANADA  (ECCC weather stations)
# ═════════════════════════════════════════════════════════════════
def find_eccc_stations(lat, lon, radius_km=150):
    """Search ECCC API for nearby stations with daily data."""
    print(f"  🔍 Searching ECCC stations within {radius_km} km…")
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    url = "https://api.weather.gc.ca/collections/climate-stations/items"
    params = {
        "f": "json", "limit": 100,
        "bbox": f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}",
        "properties": ("STATION_NAME,PROVINCE_CODE,LATITUDE,LONGITUDE,"
                       "STATION_ID,CLIMATE_ID,DLY_FIRST_DATE,DLY_LAST_DATE"),
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  ⚠️ ECCC API HTTP {r.status_code}")
            return []
        features = r.json().get("features", [])
    except Exception as e:
        print(f"  ⚠️ ECCC API: {e}")
        return []

    stations = []
    for feat in features:
        p = feat.get("properties", {})
        dly_first = p.get("DLY_FIRST_DATE")
        dly_last = p.get("DLY_LAST_DATE")
        if not dly_first or not dly_last:
            continue
        slat = p.get("LATITUDE")
        slon = p.get("LONGITUDE")
        if slat is None or slon is None:
            continue
        sid = p.get("STATION_ID") or p.get("CLIMATE_ID")
        if not sid:
            continue
        stations.append({
            "name": p.get("STATION_NAME", "Unknown"),
            "lat": slat, "lon": slon,
            "station_id": sid,
            "climate_id": p.get("CLIMATE_ID", ""),
            "first": dly_first[:10],
            "last": dly_last[:10],
            "dist_km": haversine_km(lat, lon, slat, slon),
        })

    stations.sort(key=lambda x: x["dist_km"])
    if stations:
        print(f"  📍 Found {len(stations)} stations:")
        for s in stations[:8]:
            print(f"     {s['dist_km']:5.1f} km | {s['name']:<35s} | "
                  f"{s['first']}→{s['last']} | ID:{s['station_id']}")
    else:
        print(f"  ⚠️ No ECCC stations within {radius_km} km")
    return stations


def download_eccc_station(station, start_yr, end_yr):
    """Download daily CSV from one ECCC station."""
    sid = station["station_id"]
    name = station["name"]
    print(f"\n  ⬇️ ECCC: {name} (ID:{sid}, {station['dist_km']:.1f} km)")

    frames = []
    for yr in range(start_yr, end_yr + 1):
        url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        params = {
            "format": "csv", "stationID": sid,
            "Year": yr, "Month": 1, "Day": 1,
            "timeframe": 2, "submit": "Download+Data",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                continue
            text = r.content.decode("utf-8-sig", errors="replace")
            if len(text) < 100 or "Date/Time" not in text:
                continue
            chunk = pd.read_csv(StringIO(text), parse_dates=["Date/Time"])
            if len(chunk) > 0:
                frames.append(chunk)
        except Exception:
            continue
        time.sleep(0.2)

    if not frames:
        print(f"  ⚠️ No data from {name}")
        return pd.DataFrame()

    df = (pd.concat(frames, ignore_index=True)
          .sort_values("Date/Time").drop_duplicates("Date/Time")
          .reset_index(drop=True))

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Date/Time"])

    col_map = [
        ("Total Precip", "precip_eccc"),
        ("Mean Temp",    "temp_mean_eccc"),
        ("Max Temp",     "temp_max_eccc"),
        ("Min Temp",     "temp_min_eccc"),
        ("Snow on Grnd", "snow_ground_eccc"),
    ]
    for eccc_pat, our_col in col_map:
        cols = [c for c in df.columns
                if eccc_pat in c and "Flag" not in c and "Date" not in c]
        if cols:
            out[our_col] = pd.to_numeric(df[cols[0]], errors="coerce")

    out = out.dropna(
        subset=[c for c in out.columns if c != "date"], how="all"
    )
    print(f"  ✅ {name}: {len(out):,} days")
    return out


def download_eccc(lat, lon, start_yr, end_yr,
                  max_stations=3, radius_km=150):
    """Find and download from nearest ECCC stations.
    Returns (DataFrame, list_of_station_info_dicts).
    """
    print(f"\n{'─' * 70}")
    print(f"  🍁 Environment Canada Stations")
    print(f"{'─' * 70}")

    stations = find_eccc_stations(lat, lon, radius_km)
    if not stations:
        print(f"  ↻ Expanding to {radius_km * 2} km…")
        stations = find_eccc_stations(lat, lon, radius_km * 2)
    if not stations:
        return pd.DataFrame(), []

    all_dfs = []
    station_info = []
    for stn in stations[:max_stations * 2]:  # try more to get enough
        sdf = download_eccc_station(stn, start_yr, end_yr)
        if not sdf.empty:
            all_dfs.append(sdf)
            station_info.append(stn)
            validate_df(sdf, f"ECCC: {stn['name']}")
        if len(all_dfs) >= max_stations:
            break

    if not all_dfs:
        print("  ❌ No ECCC data.")
        return pd.DataFrame(), []

    # Gap-fill from secondary stations
    merged = all_dfs[0].copy()
    for i, extra in enumerate(all_dfs[1:], 1):
        for col in [c for c in extra.columns if c != "date"]:
            if col in merged.columns:
                before = merged[col].isna().sum()
                tmp = (extra[["date", col]]
                       .drop_duplicates("date")
                       .rename(columns={col: "__f"}))
                merged = merged.merge(tmp, on="date", how="left")
                merged[col] = merged[col].fillna(merged["__f"])
                merged = merged.drop(columns=["__f"], errors="ignore")
                filled = before - merged[col].isna().sum()
                if filled > 0:
                    print(f"     +{filled:,} gap-filled {col} from "
                          f"{station_info[i]['name']}")
            else:
                try:
                    tmp = extra[["date", col]].drop_duplicates("date")
                    merged = merged.merge(tmp, on="date", how="left")
                except Exception:
                    pass

    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"\n  ✅ ECCC: {len(merged):,} days from "
          f"{len(all_dfs)} station(s)")
    print(f"     Primary: {station_info[0]['name']} "
          f"({station_info[0]['dist_km']:.1f} km)")
    validate_df(merged, "ECCC (merged)")
    return merged, station_info

# ═════════════════════════════════════════════════════════════════
# MERGE ALL SOURCES
# ═════════════════════════════════════════════════════════════════
def safe_merge(era5, power, imerg, eccc):
    avail = [(df, nm) for df, nm in
             [(era5, "ERA5"), (power, "POWER"),
              (imerg, "IMERG"), (eccc, "ECCC")]
             if not df.empty]
    if not avail:
        print("❌ No data from any source.")
        sys.exit(1)

    # Pick base dataframe
    if not era5.empty:
        df = era5.copy()
    elif not power.empty:
        df = power.copy()
        for o, n in [("_pw_temp_mean", "temp_mean"),
                     ("_pw_temp_max", "temp_max"),
                     ("_pw_temp_min", "temp_min"),
                     ("_pw_wind_max", "wind_max"),
                     ("_pw_solar_rad", "solar_rad")]:
            if o in df.columns:
                df[n] = df.pop(o)
    else:
        df = avail[0][0].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Add POWER vars
    if not power.empty and not era5.empty:
        p = power.copy()
        p["date"] = pd.to_datetime(p["date"])
        added = 0
        for col in p.columns:
            if col == "date" or col.startswith("_pw_") or col in df.columns:
                continue
            try:
                tmp = p[["date", col]].drop_duplicates("date")
                df = df.merge(tmp, on="date", how="left")
                added += 1
            except Exception:
                pass
        if added:
            print(f"    → +{added} POWER vars")

        # Fill ERA5 gaps from POWER
        for ec, pc in [("temp_mean", "_pw_temp_mean"),
                       ("temp_max", "_pw_temp_max"),
                       ("temp_min", "_pw_temp_min"),
                       ("wind_max", "_pw_wind_max"),
                       ("solar_rad", "_pw_solar_rad")]:
            if pc not in p.columns:
                continue
            try:
                tmp = (p[["date", pc]].drop_duplicates("date")
                       .rename(columns={pc: "__f"}))
                df = df.merge(tmp, on="date", how="left")
                if ec in df.columns:
                    nb = df[ec].isna().sum()
                    df[ec] = df[ec].fillna(df["__f"])
                    nf = nb - df[ec].isna().sum()
                    if nf:
                        print(f"    → Filled {nf} gaps in {ec}")
                else:
                    df[ec] = df["__f"]
                df = df.drop(columns=["__f"], errors="ignore")
            except Exception:
                df = df.drop(columns=["__f"], errors="ignore")

    # Add IMERG
    if not imerg.empty and "precip_imerg" in imerg.columns:
        im = imerg.copy()
        im["date"] = pd.to_datetime(im["date"])
        im_valid = im[im["precip_imerg"].notna()]
        if len(im_valid) > 0 and "precip_imerg" not in df.columns:
            try:
                tmp = im_valid[["date", "precip_imerg"]].drop_duplicates("date")
                df = df.merge(tmp, on="date", how="left")
                print(f"    → +precip_imerg "
                      f"({df['precip_imerg'].dropna().shape[0]:,})")
            except Exception as e:
                print(f"    ⚠️ IMERG merge: {e}")

    # Add ECCC
    if not eccc.empty:
        ec = eccc.copy()
        ec["date"] = pd.to_datetime(ec["date"])
        added = 0
        for col in ec.columns:
            if col == "date" or col in df.columns:
                continue
            try:
                tmp = ec[["date", col]].drop_duplicates("date")
                df = df.merge(tmp, on="date", how="left")
                added += 1
            except Exception:
                pass
        if added:
            print(f"    → +{added} ECCC vars")

    # Cleanup
    drop = [c for c in df.columns
            if c.startswith("_pw_") or c.startswith("__")]
    df = df.drop(columns=drop, errors="ignore")
    df = (df.loc[:, ~df.columns.duplicated()]
          .sort_values("date").reset_index(drop=True))
    av = [v for v in VMETA if v in df.columns and df[v].dropna().shape[0] > 0]
    print(f"\n  ✅ Merged: {len(df):,} days, {len(av)} variables")
    return df

# ═════════════════════════════════════════════════════════════════
# PROCESSING
# ═════════════════════════════════════════════════════════════════
def add_seasons(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(get_season)
    df["water_year"] = np.where(df["month"] >= 10,
                                df["year"] + 1, df["year"])
    return df

def add_anomalies(df, var, win=15):
    if not safe_col(df, var) or df[var].dropna().shape[0] < 60:
        return df
    try:
        df = df.copy()
        cl = df.groupby("doy")[var].agg(["mean", "std"]).reset_index()
        cl.columns = ["doy", f"{var}_cmean", f"{var}_cstd"]
        cl[f"{var}_cmean"] = (cl[f"{var}_cmean"]
                              .rolling(win, center=True, min_periods=1).mean())
        cl[f"{var}_cstd"] = (cl[f"{var}_cstd"]
                             .rolling(win, center=True, min_periods=1).mean()
                             .replace(0, np.nan))
        df = df.merge(cl, on="doy", how="left")
        df[f"{var}_anom"] = df[var] - df[f"{var}_cmean"]
        df[f"{var}_z"] = df[f"{var}_anom"] / df[f"{var}_cstd"]
    except Exception:
        pass
    return df

def get_extremes(df, var, thr=2.0):
    zc = f"{var}_z"
    if zc not in df.columns:
        return pd.DataFrame()
    try:
        ext = df[df[zc].abs() >= thr].copy()
        keep = [c for c in
                ["date", "season", "water_year", var, f"{var}_cmean", zc]
                if c in ext.columns]
        return ext[keep].sort_values(zc, ascending=False)
    except Exception:
        return pd.DataFrame()

# ═════════════════════════════════════════════════════════════════
# PLOT BUILDERS
# ═════════════════════════════════════════════════════════════════
def _av(df, vl):
    return [v for v in vl if safe_col(df, v)]

def _hover(var):
    m = VMETA.get(var, {})
    return (
        f"<b>{m.get('name', var)}</b><br>"
        f"<span style='font-size:10px'>{m.get('src_full', '')}</span><br>"
        f"%{{x|%Y-%m-%d}}: %{{y:.2f}} {m.get('unit', '')}<br>"
        f"<span style='color:#888;font-size:9px'>"
        f"{m.get('quality', '')}</span><extra></extra>"
    )

PL = dict(
    template="plotly_white", hovermode="x unified",
    hoverlabel=dict(namelength=-1, font=dict(size=11)),
)


def build_dashboard(df, varlist, title_prefix, lat, lon):
    pv = _av(df, varlist)
    if not pv:
        return None
    n = len(pv)
    try:
        titles = []
        for v in pv:
            m = VMETA.get(v, {})
            star = "★ " if m.get("quality", "").startswith("★") else ""
            titles.append(
                f"{star}{m.get('name', v)} ({m.get('unit', '')})  ⟵ "
                f"{m.get('src_full', '')}"
            )
        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True,
            vertical_spacing=max(0.004, 0.08 / n),
            subplot_titles=titles,
        )
        for i, var in enumerate(pv, 1):
            m = VMETA.get(var, {"color": "#555", "unit": ""})
            fig.add_trace(
                go.Scattergl(
                    x=df["date"], y=df[var], mode="lines",
                    name=m.get("name", var),
                    line=dict(color=m["color"], width=0.7),
                    hovertemplate=_hover(var),
                ),
                row=i, col=1,
            )
            cm = f"{var}_cmean"
            if cm in df.columns:
                fig.add_trace(
                    go.Scattergl(
                        x=df["date"], y=df[cm], mode="lines",
                        line=dict(color="#ccc", width=1, dash="dot"),
                        opacity=0.5, showlegend=False, hoverinfo="skip",
                    ),
                    row=i, col=1,
                )
            fig.update_yaxes(
                title_text=m.get("unit", ""),
                title_font=dict(size=9), row=i, col=1,
            )
        fig.update_xaxes(
            rangeselector=dict(buttons=[
                dict(count=1, label="1yr", step="year", stepmode="backward"),
                dict(count=5, label="5yr", step="year", stepmode="backward"),
                dict(count=10, label="10yr", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ], bgcolor="#eee", font=dict(size=10)),
            row=1, col=1,
        )
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.02), row=n, col=1,
        )
        fig.update_layout(
            **PL, height=170 * n + 140, showlegend=False,
            title=dict(
                text=(f"{title_prefix}<br><sup>{loc_display(lat, lon)} | "
                      f"{df['date'].min().date()}→{df['date'].max().date()}"
                      f" | ❄️Snow 💧Melt ☀️Rainfall</sup>"),
                font=dict(size=14),
            ),
            margin=dict(l=65, r=25, t=110, b=40),
        )
        return fig
    except Exception as e:
        print(f"    ⚠️ Dashboard: {e}")
        return None


def build_anomaly(df, var, lat, lon, thr=2.0):
    if not safe_col(df, var) or f"{var}_z" not in df.columns:
        return None
    m = VMETA.get(var, {})
    try:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=[
                f"{m.get('name', var)} vs Climatology",
                f"Z-Score (±{thr}σ)",
            ],
        )
        fig.add_trace(
            go.Scattergl(
                x=df["date"], y=df[var], mode="lines", name="Daily",
                line=dict(color=m.get("color", "#555"), width=0.8),
            ), row=1, col=1,
        )
        cm = f"{var}_cmean"
        if cm in df.columns:
            fig.add_trace(
                go.Scattergl(
                    x=df["date"], y=df[cm], mode="lines", name="Clim",
                    line=dict(color="#999", width=1.5, dash="dot"),
                ), row=1, col=1,
            )
        z = df[f"{var}_z"].fillna(0).values
        cols = np.where(z >= thr, "#e74c3c",
                        np.where(z <= -thr, "#3498db", "#ddd"))
        fig.add_trace(
            go.Bar(x=df["date"], y=df[f"{var}_z"], name="Z",
                   marker_color=cols.tolist(), opacity=0.8),
            row=2, col=1,
        )
        for t in [thr, -thr]:
            fig.add_hline(y=t, line_dash="dash",
                          line_color="#c0392b", opacity=0.4, row=2, col=1)
        fig.update_layout(
            **PL, height=700, showlegend=True,
            title=(f"Anomaly — {m.get('name', var)} "
                   f"[{m.get('src_full', '')}]<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
        )
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.03), row=2, col=1,
        )
        fig.update_yaxes(title_text=m.get("unit", ""), row=1, col=1)
        fig.update_yaxes(title_text="σ", row=2, col=1)
        return fig
    except Exception as e:
        print(f"    ⚠️ Anomaly {var}: {e}")
        return None


def build_seasonal(df, var, lat, lon):
    if not safe_col(df, var):
        return None
    m = VMETA.get(var, {})
    try:
        agg = (df.dropna(subset=[var])
               .groupby(["water_year", "season"])
               .agg(mean_v=(var, "mean"),
                    max_v=(var, "max"),
                    total_v=(var, "sum"))
               .reset_index())
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[f"Mean — {m.get('name', var)}", "Max", "Total"],
        )
        sc = {s: SEASONS[s]["color"] for s in SEASON_ORDER}
        for sn in SEASON_ORDER:
            ss = agg[agg["season"] == sn]
            em = SEASONS[sn]["emoji"]
            fig.add_trace(go.Bar(x=ss["water_year"], y=ss["mean_v"],
                                 name=f"{em} {sn}", marker_color=sc[sn],
                                 opacity=0.8, legendgroup=sn), row=1, col=1)
            fig.add_trace(go.Bar(x=ss["water_year"], y=ss["max_v"],
                                 name=f"{em} {sn}", marker_color=sc[sn],
                                 opacity=0.8, legendgroup=sn,
                                 showlegend=False), row=2, col=1)
            fig.add_trace(go.Bar(x=ss["water_year"], y=ss["total_v"],
                                 name=f"{em} {sn}", marker_color=sc[sn],
                                 opacity=0.8, legendgroup=sn,
                                 showlegend=False), row=3, col=1)
        fig.update_layout(
            **PL, height=900, barmode="group",
            title=(f"3-Season — {m.get('name', var)} ({m.get('unit', '')})"
                   f" [{m.get('src_full', '')}]<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
            legend=dict(orientation="h", y=1.02, xanchor="center", x=0.5),
        )
        for r in range(1, 4):
            fig.update_yaxes(title_text=m.get("unit", ""), row=r, col=1)
        return fig
    except Exception as e:
        print(f"    ⚠️ Seasonal {var}: {e}")
        return None


def build_heatmap(df, var, lat, lon):
    if not safe_col(df, var):
        return None
    m = VMETA.get(var, {})
    try:
        mo = (df.dropna(subset=[var])
              .groupby(["year", "month"])[var].mean().reset_index())
        piv = mo.pivot(index="year", columns="month", values=var)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if "temp" in var:
            cs = "RdYlBu_r"
        elif "soil" in var:
            cs = "BrBG"
        else:
            cs = "YlGnBu"
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=months, y=piv.index, colorscale=cs,
            colorbar=dict(title=m.get("unit", "")),
            hovertemplate=("Year %{y}, %{x}: %{z:.2f} "
                           + m.get("unit", "") + "<extra></extra>"),
        ))
        fig.update_layout(
            **PL,
            title=(f"Heatmap — {m.get('name', var)} "
                   f"[{m.get('src_full', '')}]<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
            xaxis_title="Month", yaxis_title="Year",
            height=max(400, len(piv) * 16 + 170),
            yaxis=dict(autorange="reversed"),
        )
        return fig
    except Exception as e:
        print(f"    ⚠️ Heatmap {var}: {e}")
        return None


def build_freeze_thaw(df, lat, lon):
    if not safe_col(df, "temp_max") or not safe_col(df, "temp_min"):
        return None
    try:
        d = df.copy()
        d["ft"] = (d["temp_min"] < 0) & (d["temp_max"] > 0)
        ft = d.groupby(["water_year", "season"])["ft"].sum().reset_index()
        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.12,
            subplot_titles=["Freeze-Thaw Days by Season",
                            "Daily Temp Envelope"],
        )
        sc = {s: SEASONS[s]["color"] for s in SEASON_ORDER}
        for sn in SEASON_ORDER:
            ss = ft[ft["season"] == sn]
            fig.add_trace(
                go.Bar(x=ss["water_year"], y=ss["ft"],
                       name=f"{SEASONS[sn]['emoji']} {sn}",
                       marker_color=sc[sn], opacity=0.8),
                row=1, col=1,
            )
        fig.add_trace(
            go.Scattergl(x=d["date"], y=d["temp_max"], mode="lines",
                         name="Tmax", line=dict(color="#c0392b", width=0.5)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=d["date"], y=d["temp_min"], mode="lines",
                         name="Tmin", line=dict(color="#2980b9", width=0.5)),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=2, col=1)
        fig.update_layout(
            **PL, height=800, barmode="group",
            title=(f"Freeze-Thaw [ERA5-Land]<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
        )
        fig.update_yaxes(title_text="Days", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.03), row=2, col=1,
        )
        return fig
    except Exception as e:
        print(f"    ⚠️ FT: {e}")
        return None


def build_precip_compare(df, lat, lon):
    srcs = _av(df, ["precip_era5", "precip_power",
                     "precip_imerg", "precip_eccc"])
    if len(srcs) < 2:
        return None
    try:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=["Daily Precipitation — All Sources",
                            "Difference from ERA5", "Annual Totals"],
        )
        cols = {"precip_era5": "#2471a3", "precip_power": "#1a5276",
                "precip_imerg": "#154360", "precip_eccc": "#117864"}
        for v in srcs:
            fig.add_trace(
                go.Scattergl(
                    x=df["date"], y=df[v], mode="lines",
                    name=VMETA[v]["name"],
                    line=dict(color=cols.get(v, "#555"), width=0.7),
                    hovertemplate=_hover(v),
                ), row=1, col=1,
            )
        pri = "precip_era5" if "precip_era5" in srcs else srcs[0]
        for v in srcs:
            if v == pri:
                continue
            diff = df[pri].fillna(0) - df[v].fillna(0)
            fig.add_trace(
                go.Scattergl(
                    x=df["date"], y=diff, mode="lines",
                    name=f"Δ ERA5−{VMETA[v]['name'][:15]}",
                    line=dict(width=0.7),
                ), row=2, col=1,
            )
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=2, col=1)
        for v in srcs:
            try:
                wy = df.groupby("water_year")[v].sum().reset_index()
                fig.add_trace(
                    go.Bar(x=wy["water_year"], y=wy[v],
                           name=VMETA[v]["name"][:20],
                           marker_color=cols.get(v, "#555"), opacity=0.7),
                    row=3, col=1,
                )
            except Exception:
                pass
        fig.update_layout(
            **PL, height=950, barmode="group",
            title=(f"Precipitation: All Sources<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
        )
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.03), row=2, col=1,
        )
        for r in [1, 2]:
            fig.update_yaxes(title_text="mm/day", row=r, col=1)
        fig.update_yaxes(title_text="mm/yr", row=3, col=1)
        return fig
    except Exception as e:
        print(f"    ⚠️ Precip: {e}")
        return None


def build_soil(df, lat, lon):
    layers = _av(df, ["soil_moist_sfc", "soil_moist_root", "soil_moist_prof"])
    if not layers:
        return None
    try:
        n = len(layers)
        titles = [f"{VMETA[v]['name']} — {VMETA[v].get('quality', '')}"
                  for v in layers]
        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, subplot_titles=titles,
        )
        for i, var in enumerate(layers, 1):
            m = VMETA[var]
            fig.add_trace(
                go.Scattergl(
                    x=df["date"], y=df[var], mode="lines",
                    name=m["name"],
                    line=dict(color=m["color"], width=0.8),
                    hovertemplate=_hover(var),
                ), row=i, col=1,
            )
            fig.update_yaxes(title_text="fraction", row=i, col=1)
        fig.update_layout(
            **PL, height=300 * n + 100, showlegend=False,
            title=(f"Soil Moisture [POWER]<br>"
                   f"<sup>{loc_display(lat, lon)}</sup>"),
        )
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.03), row=n, col=1,
        )
        return fig
    except Exception as e:
        print(f"    ⚠️ Soil: {e}")
        return None


def build_quality(df, lat, lon):
    avail = [v for v in VMETA if safe_col(df, v)]
    if not avail:
        return None
    try:
        rd = []
        for v in avail:
            m = VMETA[v]
            n = df[v].dropna().shape[0]
            q = m.get("quality", "")
            rd.append({
                "var": m["name"], "source": m["src"],
                "sf": m["src_full"], "n": n, "q": q,
                "best": "★" in q,
            })
        tdf = pd.DataFrame(rd)
        fig = go.Figure()
        for src, color in [("ERA5Land", "#2471a3"), ("POWER", "#8e44ad"),
                           ("IMERG", "#154360"), ("ECCC", "#117864")]:
            sd = tdf[tdf["source"] == src]
            if sd.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sd["n"], y=sd["var"], mode="markers+text", name=src,
                marker=dict(
                    color=color,
                    size=[14 if b else 8 for b in sd["best"]],
                    symbol=["star" if b else "circle" for b in sd["best"]],
                ),
                text=["★" if b else "" for b in sd["best"]],
                textposition="middle right",
                hovertemplate=[
                    f"<b>{r['var']}</b><br>{r['sf']}<br>"
                    f"{r['n']:,} days<br><i>{r['q']}</i><extra></extra>"
                    for _, r in sd.iterrows()
                ],
            ))
        fig.update_layout(
            **PL, height=max(500, len(avail) * 26 + 150),
            title=(f"Source Quality<br>"
                   f"<sup>★=Recommended | {loc_display(lat, lon)}</sup>"),
            xaxis_title="Days",
            legend=dict(orientation="h", y=1.05),
        )
        return fig
    except Exception as e:
        print(f"    ⚠️ Quality: {e}")
        return None

# ═════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════
def save_plot(fig, fp):
    if fig is None:
        return False
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(fp), include_plotlyjs="cdn")
        print(f"  ✅ {fp.name}")
        return True
    except Exception as e:
        print(f"  ❌ {fp.name}: {e}")
        return False


def save_data(df, era5, power, imerg, eccc,
              lat, lon, raw_dir, proc_dir, zth):
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    lt = loc_tag(lat, lon)

    print(f"\n  Raw → {raw_dir}/")
    for sdf, pfx in [(era5, "ERA5Land"), (power, "POWER"),
                     (imerg, "IMERG"), (eccc, "ECCC")]:
        if not sdf.empty:
            fp = raw_dir / f"{pfx}_daily_{lt}.csv"
            sdf.to_csv(fp, index=False, float_format="%.4f")
            dcols = [c for c in sdf.columns if c != "date"]
            nn = sdf[dcols].notna().any(axis=1).sum() if dcols else 0
            print(f"  💾 {fp.name} ({len(sdf):,} rows, {nn:,} with data)")

    print(f"\n  Processed → {proc_dir}/")
    fp = proc_dir / f"MERGED_daily_{lt}.csv"
    df.to_csv(fp, index=False, float_format="%.4f")
    print(f"  💾 {fp.name} ({len(df):,}×{len(df.columns)})")

    avars = [v for v in VMETA
             if safe_col(df, v) and f"{v}_z" in df.columns]
    ext_all = []
    for v in avars:
        e = get_extremes(df, v, zth)
        if not e.empty:
            e["variable"] = v
            e["source"] = VMETA[v].get("src_full", "")
            ext_all.append(e)
    if ext_all:
        c = pd.concat(ext_all, ignore_index=True)
        fp = proc_dir / f"MERGED_extremes_{lt}.csv"
        c.to_csv(fp, index=False, float_format="%.4f")
        print(f"  💾 {fp.name} ({len(c):,})")

# ═════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════
# INDEX.HTML
# ═════════════════════════════════════════════════════════════════
def build_index(outdir, lat, lon, start_yr, end_yr,
                registry, has_imerg, has_eccc, eccc_stations=None):
    """Generate index.html dashboard linking all plots."""
    sec_order = ["dashboard", "precipitation", "temperature",
                 "snow_soil", "wind_atm", "analysis", "quality"]
    sec_labels = {
        "dashboard":     ("Dashboards",        "Multi-variable time-series by source"),
        "precipitation": ("Precipitation",      "Comparison, seasonal, anomaly, heatmap"),
        "temperature":   ("Temperature",        "Seasonal, anomaly, heatmap"),
        "snow_soil":     ("Snow & Soil",        "Snow depth, soil moisture profiles"),
        "wind_atm":      ("Wind & Atmosphere",  "Wind, solar, ET0, humidity, pressure"),
        "analysis":      ("Analysis",           "Freeze-thaw cycles"),
        "quality":       ("Data Quality",       "Coverage and recommended sources"),
    }
    grouped = {s: [] for s in sec_order}
    for p in registry:
        grouped.get(p.get("section", "analysis"), grouped["analysis"]).append(p)

    sources = ["ERA5-Land 9 km", "POWER 50 km"]
    if has_imerg:
        sources.append("IMERG 10 km")
    if has_eccc:
        sources.append("ECCC Stations")

    shtml = ""
    for sec in sec_order:
        items = grouped[sec]
        if not items:
            continue
        label, desc = sec_labels[sec]
        shtml += (
            f'<section class="sec">'
            f'<div class="sh"><h2 class="st">{label}</h2>'
            f'<span class="sd">{desc}</span></div>'
            f'<div class="sg">'
        )
        for item in items:
            src = item.get("source", "")
            bc = {"ERA5Land": "era5", "POWER": "power", "IMERG": "imerg",
                  "ECCC": "eccc", "Combined": "comb"}.get(src, "comb")
            bl = {"ERA5Land": "ERA5-Land", "POWER": "POWER",
                  "IMERG": "IMERG", "ECCC": "ECCC",
                  "Combined": "Combined"}.get(src, src)
            shtml += (
                f'<a href="plots/{item["file"]}" target="_blank" class="pc">'
                f'<span class="pt">{item["title"]}</span>'
                f'<span class="b {bc}">{bl}</span>'
                f'</a>'
            )
        shtml += "</div></section>"

    eccc_html = ""
    if has_eccc and eccc_stations:
        rows = ""
        for i, stn in enumerate(eccc_stations):
            role = "Primary" if i == 0 else "Gap-fill"
            rows += (
                f'<tr><td>{role}</td><td>{stn["name"]}</td>'
                f'<td>{stn["dist_km"]:.1f} km</td>'
                f'<td>{stn["first"]} &mdash; {stn["last"]}</td>'
                f'<td>{stn["station_id"]}</td></tr>'
            )
        eccc_html = (
            '<section class="sec">'
            '<div class="sh"><h2 class="st">ECCC Stations Used</h2></div>'
            '<table class="ref-tbl"><thead>'
            '<tr><th>Role</th><th>Station</th><th>Distance</th>'
            '<th>Record</th><th>ID</th></tr>'
            f'</thead><tbody>{rows}</tbody></table></section>'
        )

    src_rows = [
        ("ERA5-Land", "ECMWF reanalysis", "9 km", "1950 &ndash; present",
         "Temp, precip, wind, solar, ET0"),
        ("NASA POWER", "MERRA-2 / GEOS / CERES", "50 km", "1981 &ndash; present",
         "Snow depth, soil moisture, humidity, pressure"),
    ]
    if has_imerg:
        src_rows.append(
            ("GPM IMERG", "Satellite + gauge", "10 km", "Jun 2000 &ndash; present",
             "Precipitation only"))
    if has_eccc:
        src_rows.append(
            ("ECCC", "Ground stations", "Point", "Varies",
             "Precip, temp, snow on ground"))
    src_html = ""
    for name, typ, res, period, vars_ in src_rows:
        src_html += (
            f'<tr><td class="src-name">{name}</td><td>{typ}</td>'
            f'<td>{res}</td><td>{period}</td><td>{vars_}</td></tr>')

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Climate Explorer &mdash; {loc_display(lat, lon)}</title>
<style>
:root{{--bg:#f5f6f8;--card:#fff;--border:#e2e5ea;--text:#1d2939;
  --muted:#667085;--accent:#1570ef;--era5:#2471a3;--power:#7c3aed;
  --imerg:#0e7490;--eccc:#059669;--comb:#475467}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,
  sans-serif;background:var(--bg);color:var(--text);line-height:1.5;
  -webkit-font-smoothing:antialiased}}
header{{background:#101828;color:#fff;padding:28px 0}}
header .wrap{{max-width:1140px;margin:0 auto;padding:0 24px;
  display:flex;justify-content:space-between;align-items:flex-start;
  flex-wrap:wrap;gap:16px}}
.hd-left h1{{font-size:18px;font-weight:700;letter-spacing:-0.2px;margin-bottom:2px}}
.hd-meta{{font-size:13px;color:#94a3b8;display:flex;flex-wrap:wrap;gap:16px}}
.hd-meta span{{white-space:nowrap}}
.seasons{{display:flex;gap:6px;margin-top:10px}}
.stag{{font-size:11px;font-weight:600;padding:3px 10px;border-radius:12px;
  letter-spacing:0.2px}}
.stag.sn{{background:rgba(59,130,246,0.18);color:#93c5fd}}
.stag.ml{{background:rgba(245,158,11,0.18);color:#fcd34d}}
.stag.rn{{background:rgba(239,68,68,0.18);color:#fca5a5}}
.wrap{{max-width:1140px;margin:0 auto;padding:0 24px}}
main{{padding:24px 0 48px}}
.sec{{margin-bottom:32px}}
.sh{{display:flex;align-items:baseline;gap:12px;margin-bottom:12px;
  border-bottom:1px solid var(--border);padding-bottom:8px}}
.st{{font-size:15px;font-weight:700;color:var(--text)}}
.sd{{font-size:12px;color:var(--muted)}}
.sg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}}
.pc{{display:flex;align-items:center;justify-content:space-between;
  background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:10px 14px;text-decoration:none;color:var(--text);
  transition:border-color .15s,box-shadow .15s}}
.pc:hover{{border-color:var(--accent);box-shadow:0 2px 8px rgba(0,0,0,0.06)}}
.pt{{font-size:13px;font-weight:500}}
.b{{font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px;
  white-space:nowrap;flex-shrink:0;margin-left:8px}}
.b.era5{{background:#eff8ff;color:var(--era5)}}
.b.power{{background:#f5f3ff;color:var(--power)}}
.b.imerg{{background:#ecfeff;color:var(--imerg)}}
.b.eccc{{background:#ecfdf5;color:var(--eccc)}}
.b.comb{{background:#f2f4f7;color:var(--comb)}}
.ref-tbl,.src-tbl{{width:100%;border-collapse:collapse;font-size:12px;
  background:var(--card);border:1px solid var(--border);border-radius:8px;
  overflow:hidden}}
.ref-tbl th,.src-tbl th{{text-align:left;padding:8px 12px;font-weight:600;
  font-size:11px;text-transform:uppercase;letter-spacing:0.4px;
  color:var(--muted);background:#f9fafb;border-bottom:1px solid var(--border)}}
.ref-tbl td,.src-tbl td{{padding:7px 12px;border-bottom:1px solid #f2f4f7}}
.ref-tbl tr:last-child td,.src-tbl tr:last-child td{{border-bottom:none}}
.src-name{{font-weight:600}}
footer{{text-align:center;padding:20px;font-size:11px;color:var(--muted);
  border-top:1px solid var(--border)}}
@media(max-width:640px){{
  header .wrap{{flex-direction:column}}
  .sg{{grid-template-columns:1fr}}
  .ref-tbl,.src-tbl{{font-size:11px}}
  .ref-tbl th,.src-tbl th,.ref-tbl td,.src-tbl td{{padding:5px 8px}}
}}
</style></head><body>
<header><div class="wrap">
<div class="hd-left">
<h1>Climate Data Explorer</h1>
<div class="hd-meta">
<span>{loc_display(lat, lon)}</span>
<span>{start_yr}&ndash;{end_yr}</span>
<span>{' &middot; '.join(sources)}</span>
</div>
<div class="seasons">
<span class="stag sn">Snow Nov&ndash;Mar</span>
<span class="stag ml">Melt Apr&ndash;May</span>
<span class="stag rn">Rainfall Jun&ndash;Oct</span>
</div>
</div>
</div></header>
<main><div class="wrap">
{shtml}
{eccc_html}
<section class="sec">
<div class="sh"><h2 class="st">Data Sources</h2></div>
<table class="src-tbl"><thead>
<tr><th>Source</th><th>Type</th><th>Resolution</th><th>Period</th><th>Variables</th></tr>
</thead><tbody>{src_html}</tbody></table>
</section>
</div></main>
<footer>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</footer>
</body></html>"""

    fp = outdir / "index.html"
    fp.write_text(html, encoding="utf-8")
    print(f"  ✅ {fp}")
# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description="Multi-source climate data explorer")
    pa.add_argument("--lat", type=float, default=DEFAULT_LAT)
    pa.add_argument("--lon", type=float, default=DEFAULT_LON)
    pa.add_argument("--start", type=int, default=2000)
    pa.add_argument("--end", type=int, default=datetime.now().year)
    pa.add_argument("--project", type=str, default="Whistler_Climate")
    pa.add_argument("--zscore", type=float, default=2.0)
    pa.add_argument("--no-plots", action="store_true")
    pa.add_argument("--no-imerg", action="store_true")
    pa.add_argument("--no-eccc", action="store_true")
    pa.add_argument("--earthdata-user", type=str, default="")
    pa.add_argument("--earthdata-pass", type=str, default="")
    pa.add_argument("--imerg-start", type=str, default="")
    pa.add_argument("--imerg-end", type=str, default="")
    pa.add_argument("--imerg-workers", type=int, default=8)
    pa.add_argument("--eccc-radius", type=int, default=150)
    pa.add_argument("--eccc-stations", type=int, default=3)
    pa.add_argument("--force-redownload", action="store_true")
    pa.add_argument("--no-prompts", action="store_true",
                    help="Skip all interactive prompts — use arg/default values")
    args = pa.parse_args()

    args.lat_set = "--lat" in sys.argv or args.no_prompts
    args.start_set = "--start" in sys.argv or args.no_prompts
    # Auto-enable IMERG (no prompt) when credentials are supplied or --no-prompts set
    has_creds = bool(args.earthdata_user) and bool(args.earthdata_pass)
    args.no_imerg_prompt = (args.no_imerg or "--imerg-start" in sys.argv
                            or has_creds or args.no_prompts)
    args.no_eccc_prompt = args.no_eccc or args.no_prompts

    global IMERG_WORKERS
    IMERG_WORKERS = max(1, min(20, args.imerg_workers))

    (lat, lon, start_yr, end_yr, project,
     do_imerg, ims, ime, do_eccc) = prompt_settings(args)
    if not ims and args.imerg_start:
        ims = args.imerg_start
    if not ime and args.imerg_end:
        ime = args.imerg_end

    # Single project folder
    outdir = Path(project)
    raw_dir = outdir / "raw"
    proc_dir = outdir / "processed"
    plots_dir = outdir / "plots"
    for d in [outdir, raw_dir, proc_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  📂 {outdir.resolve()}")
    print(f"     raw/ · processed/ · plots/")
    print(f"{'=' * 70}")

    # ── DOWNLOAD ──
    era5_df = pd.DataFrame()
    power_df = pd.DataFrame()
    imerg_df = pd.DataFrame()
    eccc_df = pd.DataFrame()
    eccc_station_info = []

    c = (check_cached(raw_dir, "ERA5Land", lat, lon)
         if not args.force_redownload else None)
    if c and ask_reuse("ERA5-Land", c):
        era5_df = load_cached(c)
    else:
        era5_df = download_era5land(lat, lon, start_yr, end_yr)

    c = (check_cached(raw_dir, "POWER", lat, lon)
         if not args.force_redownload else None)
    if c and ask_reuse("POWER", c):
        power_df = load_cached(c)
    else:
        power_df = download_power(lat, lon, start_yr, end_yr)

    if do_imerg:
        c = (check_cached(raw_dir, "IMERG", lat, lon)
             if not args.force_redownload else None)
        if c and ask_reuse("IMERG", c, default_yes=True):
            imerg_df = load_cached(c)
        else:
            imerg_df = download_imerg(
                lat, lon, start_yr, end_yr, raw_dir,
                args.earthdata_user, args.earthdata_pass, ims, ime,
            )

    if do_eccc:
        c = (check_cached(raw_dir, "ECCC", lat, lon)
             if not args.force_redownload else None)
        if c and ask_reuse("ECCC Stations", c, default_yes=True):
            eccc_df = load_cached(c)
            # Try to load saved station info
            si_fp = raw_dir / f".eccc_stations_{loc_tag(lat, lon)}.json"
            if si_fp.exists():
                try:
                    eccc_station_info = json.loads(si_fp.read_text())
                except Exception:
                    eccc_station_info = []
        else:
            eccc_df, eccc_station_info = download_eccc(
                lat, lon, start_yr, end_yr,
                max_stations=args.eccc_stations,
                radius_km=args.eccc_radius,
            )
            # Save station info for cache reuse
            if eccc_station_info:
                si_fp = raw_dir / f".eccc_stations_{loc_tag(lat, lon)}.json"
                try:
                    si_fp.write_text(json.dumps(eccc_station_info,
                                                indent=2, default=str))
                except Exception:
                    pass

    # ── MERGE ──
    print(f"\n{'─' * 70}\n  MERGING\n{'─' * 70}")
    df = safe_merge(era5_df, power_df, imerg_df, eccc_df)
    has_imerg = not imerg_df.empty and safe_col(df, "precip_imerg")
    has_eccc = (not eccc_df.empty and
                any(safe_col(df, v) for v in
                    ["precip_eccc", "temp_mean_eccc"]))

    # ── PROCESS ──
    print("\n🔧 Seasons + anomalies…")
    df = add_seasons(df)
    for var in VMETA:
        if safe_col(df, var) and df[var].dropna().shape[0] > 60:
            df = add_anomalies(df, var)

    # ── SAVE ──
    print(f"\n{'─' * 70}\n  SAVING\n{'─' * 70}")
    save_data(df, era5_df, power_df, imerg_df, eccc_df,
              lat, lon, raw_dir, proc_dir, args.zscore)

    if args.no_plots:
        print("\n⏭️ Skip plots.")
        return

    # ── PLOTS ──
    print(f"\n{'─' * 70}\n  PLOTS\n{'─' * 70}")
    reg = []

    def sp(fig, fn, sec, title, desc="", src="", star=False):
        if save_plot(fig, plots_dir / fn):
            reg.append({"file": fn, "section": sec, "title": title,
                        "desc": desc, "source": src, "star": star})

    # ─── Dashboards ───
    print("\n  [Dashboards]")
    sp(build_dashboard(df, DASH_COMBINED,
                       "Combined — All Sources", lat, lon),
       "Combined_dashboard.html", "dashboard",
       "Combined Dashboard", "All variables", "Combined", True)
    if not era5_df.empty:
        sp(build_dashboard(df, DASH_ERA5,
                           "ERA5-Land (9 km)", lat, lon),
           "ERA5Land_dashboard.html", "dashboard",
           "ERA5-Land Dashboard", "", "ERA5Land")
    if not power_df.empty:
        sp(build_dashboard(df, DASH_POWER,
                           "POWER (50 km)", lat, lon),
           "POWER_dashboard.html", "dashboard",
           "POWER Dashboard", "", "POWER")
    if has_imerg:
        sp(build_dashboard(df, DASH_IMERG,
                           "GPM IMERG (10 km satellite)", lat, lon),
           "IMERG_dashboard.html", "dashboard",
           "IMERG Dashboard", "Satellite precip", "IMERG", True)
    if has_eccc:
        sp(build_dashboard(df, DASH_ECCC,
                           "ECCC Stations (ground truth)", lat, lon),
           "ECCC_dashboard.html", "dashboard",
           "ECCC Station Dashboard", "Ground truth", "ECCC", True)

    # ─── Precipitation ───
    print("\n  [Precipitation]")
    sp(build_precip_compare(df, lat, lon),
       "Precip_comparison.html", "precipitation",
       "All Sources Compared", "", "Combined", True)
    for v in ["precip_era5", "precip_power", "precip_imerg", "precip_eccc"]:
        if not safe_col(df, v):
            continue
        src = VMETA[v]["src"]
        sn = VMETA[v]["name"]
        sp(build_seasonal(df, v, lat, lon),
           f"Precip_seasonal_{src}.html", "precipitation",
           f"{sn} — 3-Season", "", src)
        sp(build_anomaly(df, v, lat, lon, args.zscore),
           f"Precip_anomaly_{src}.html", "precipitation",
           f"{sn} — Anomalies", "", src)
        sp(build_heatmap(df, v, lat, lon),
           f"Precip_heatmap_{src}.html", "precipitation",
           f"{sn} — Heatmap", "", src)

    # ─── Temperature ───
    print("\n  [Temperature]")
    for v in ["temp_mean", "temp_max", "temp_min",
              "temp_mean_eccc", "temp_max_eccc", "temp_min_eccc"]:
        if not safe_col(df, v):
            continue
        m = VMETA.get(v, {})
        vn = m.get("name", v)
        src = m.get("src", "ERA5Land")
        sp(build_seasonal(df, v, lat, lon),
           f"Temp_seasonal_{v}.html", "temperature",
           f"{vn} — 3-Season", "", src)
        sp(build_anomaly(df, v, lat, lon, args.zscore),
           f"Temp_anomaly_{v}.html", "temperature",
           f"{vn} — Anomalies", "", src)
        sp(build_heatmap(df, v, lat, lon),
           f"Temp_heatmap_{v}.html", "temperature",
           f"{vn} — Heatmap", "", src)

    # ─── Snow & Soil ───
    print("\n  [Snow & Soil]")
    sp(build_soil(df, lat, lon),
       "Soil_moisture_3layer.html", "snow_soil",
       "Soil Moisture — 3 Layers", "", "POWER", True)
    for v in ["snow_depth", "snow_ground_eccc",
              "soil_moist_sfc", "soil_moist_root", "soil_moist_prof"]:
        if not safe_col(df, v):
            continue
        m = VMETA.get(v, {})
        vn = m.get("name", v)
        src = m.get("src", "POWER")
        sp(build_seasonal(df, v, lat, lon),
           f"SnowSoil_seasonal_{v}.html", "snow_soil",
           f"{vn} — 3-Season", "", src)
        sp(build_anomaly(df, v, lat, lon, args.zscore),
           f"SnowSoil_anomaly_{v}.html", "snow_soil",
           f"{vn} — Anomalies", "", src)
        sp(build_heatmap(df, v, lat, lon),
           f"SnowSoil_heatmap_{v}.html", "snow_soil",
           f"{vn} — Heatmap", "", src)

    # ─── Wind & Atmosphere ───
    print("\n  [Wind & Atmosphere]")
    for v in ["wind_max", "wind_gust", "solar_rad", "et0",
              "humidity", "dewpoint", "pressure", "cloud_cover"]:
        if not safe_col(df, v):
            continue
        m = VMETA[v]
        vn = m["name"]
        src = m["src"]
        sp(build_seasonal(df, v, lat, lon),
           f"Atm_seasonal_{v}.html", "wind_atm",
           f"{vn} — 3-Season", "", src)
        sp(build_anomaly(df, v, lat, lon, args.zscore),
           f"Atm_anomaly_{v}.html", "wind_atm",
           f"{vn} — Anomalies", "", src)
        sp(build_heatmap(df, v, lat, lon),
           f"Atm_heatmap_{v}.html", "wind_atm",
           f"{vn} — Heatmap", "", src)

    # ─── Analysis ───
    print("\n  [Analysis]")
    sp(build_freeze_thaw(df, lat, lon),
       "Freeze_thaw.html", "analysis",
       "Freeze-Thaw", "", "ERA5Land", True)
    sp(build_quality(df, lat, lon),
       "Source_quality.html", "quality",
       "Source Quality", "", "Combined", True)

    # ── INDEX ──
    build_index(outdir, lat, lon, start_yr, end_yr,
                reg, has_imerg, has_eccc, eccc_station_info)

    # ── SUMMARY ──
    np_ = len(reg)
    print(f"\n{'=' * 60}")
    print(f"  Done — {np_} plots")
    print(f"  {outdir.resolve()}/index.html")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()