# Climate Data Explorer

Downloads daily climate data from multiple sources and generates interactive
Plotly dashboards with three-season hydrological classification.

**Sources:** ERA5-Land (9 km), NASA POWER (50 km), GPM IMERG (10 km),
Environment Canada (ECCC) weather stations.

## Usage

```bash
pip install requests pandas numpy plotly
python climate_explorer_web.py
```

The script prompts for coordinates, date range, and which sources to include.
To skip prompts:

```bash
python climate_explorer_web.py --lat 49.5 --lon -121.3 \
    --start 2000 --end 2025 --project MySite --no-imerg
```

Then open `MySite/index.html` in a browser.

## Options

| Flag                 | Description              | Default          |
| -------------------- | ------------------------ | ---------------- |
| `--lat`, `--lon`     | Site coordinates         | 56.1119, -120.78 |
| `--start`, `--end`   | Year range               | 1990 to current  |
| `--project`          | Output folder            | prompted         |
| `--no-imerg`         | Skip IMERG               | off              |
| `--no-eccc`          | Skip ECCC stations       | off              |
| `--imerg-start/end`  | IMERG date range         | prompted         |
| `--imerg-workers`    | Parallel threads         | 8                |
| `--eccc-radius`      | Station search radius km | 150              |
| `--eccc-stations`    | Max stations to use      | 3                |
| `--zscore`           | Anomaly threshold        | 2.0              |
| `--no-plots`         | Data only, skip plots    | off              |
| `--force-redownload` | Ignore cached files      | off              |
| `--earthdata-user`   | NASA Earthdata username  | prompted/env     |
| `--earthdata-pass`   | NASA Earthdata password  | prompted/env     |

## Output

```
MySite/
├── index.html           <- Dashboard (open in browser)
├── plots/               <- 40-60+ interactive HTML plots
├── raw/                 <- Per-source CSVs + cache files
└── processed/           <- Merged daily CSV + extremes
```

## Data Sources

| Source     | Resolution | Period            | Key Variables                           |
| ---------- | ---------- | ----------------- | --------------------------------------- |
| ERA5-Land  | 9 km       | 1950 - present    | Temp, precip, wind, solar, ET0          |
| NASA POWER | 50 km      | 1981 - present    | Snow depth, soil moisture, humidity     |
| GPM IMERG  | 10 km      | Jun 2000 - present| Precipitation only (Earthdata account)  |
| ECCC       | Point      | Varies by station | Precip, temp, snow on ground            |

## Three-Season Classification

| Season   | Months  | Meaning                                        |
| -------- | ------- | ---------------------------------------------- |
| Snow     | Nov-Mar | Accumulation, frozen ground, low baseflow      |
| Melt     | Apr-May | Snowmelt, rising pore-pressure, landslide risk |
| Rainfall | Jun-Oct | Rain-dominated, high ET, convective storms     |

Water year starts October 1.

## Caching

Downloaded data is cached per source. IMERG and ECCC default to reuse
(slow to download). ERA5-Land and POWER default to re-download (fast).
Use `--force-redownload` to ignore all caches.

IMERG supports resume — if interrupted, re-running picks up where it stopped.
Failed days are automatically retried in a gap-fill pass.

## ECCC Stations

The script searches for weather stations within `--eccc-radius` km, downloads
from the nearest ones, and gap-fills across up to 3 stations. Station metadata
(name, distance, record period) is shown in the dashboard.

## IMERG Notes

- Requires a free NASA Earthdata account with GES DISC access approved
  (https://disc.gsfc.nasa.gov/earthdata-login)
- Uses the Final V07B product with gauge calibration
- Each day retries up to 3 times; a gap-fill pass re-attempts failed days
- At high latitudes (>60N), satellite retrieval may underestimate frozen precip

## Troubleshooting

- **ERA5 hangs:** Archive API can't serve future dates. Try `--end 2025`
- **IMERG auth fails:** Approve GES DISC at the link above
- **No ECCC stations:** Increase `--eccc-radius` (e.g. 300)
- **Rate limits:** Built-in retry with backoff. Wait 5 min if persistent

## Requirements

Python 3.8+, `requests>=2.28`, `pandas>=1.5`, `numpy>=1.23`, `plotly>=5.14`
