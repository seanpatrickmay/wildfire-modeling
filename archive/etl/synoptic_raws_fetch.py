#!/usr/bin/env python3
"""
Fetch RAWS stations and hourly data from Synoptic Data API.

Example:
  SYNOPTIC_TOKEN=YOUR_TOKEN python3 scripts/synoptic_raws_fetch.py \
    --bbox -122.96 38.50 -122.59 38.87 \
    --start 201910240000 --end 201910302300 \
    --output data/raws/kincade
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


BASE_URL = "https://api.synopticdata.com/v2/"
ENV_KEYS = (
    "SYNOPTIC_TOKEN",
    "SYNOPTIC_API_TOKEN",
    "SYNOPTIC_PUBLIC_TOKEN",
    "TOKEN",
    "token",
)


def load_dotenv(path: str = ".env") -> dict:
    if not os.path.exists(path):
        return {}
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            env[key] = value
    return env


def get_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    for key in ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    dotenv = load_dotenv()
    for key in ENV_KEYS:
        value = dotenv.get(key)
        if value:
            return value
    raise SystemExit(
        "Missing Synoptic token. Set SYNOPTIC_TOKEN (or TOKEN) or pass --token."
    )


def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def get_raws_network_ids(token: str) -> list[str]:
    url = f"{BASE_URL}networks?token={urllib.parse.quote(token)}"
    data = http_get_json(url)
    networks = data.get("MNET", [])
    raws_ids = []
    for net in networks:
        name = (net.get("NAME") or "").lower()
        short = (net.get("SHORTNAME") or "").lower()
        if "raws" in name or "raws" in short:
            raws_ids.append(str(net.get("ID")))
    return raws_ids


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def hour_key(iso_time: str) -> str:
    # Expect ISO 8601 with Z
    return iso_time[:13] + ":00:00Z"


def circular_mean(deg_list):
    import math

    if not deg_list:
        return None
    rad = [math.radians(x) for x in deg_list]
    sin_sum = sum(math.sin(x) for x in rad)
    cos_sum = sum(math.cos(x) for x in rad)
    if sin_sum == 0 and cos_sum == 0:
        return None
    angle = math.degrees(math.atan2(sin_sum, cos_sum))
    return angle % 360.0


def aggregate_hourly(observations: dict, time_steps: list[str]) -> dict:
    # Build per-variable hourly aggregates
    # observations: {var: [values], date_time: [iso]}
    date_times = observations.get("date_time", [])
    if not date_times:
        return {}
    # Build mapping hour -> indices
    hour_map = {}
    for idx, dt in enumerate(date_times):
        hour = hour_key(dt)
        hour_map.setdefault(hour, []).append(idx)

    result = {}
    for var, values in observations.items():
        if var == "date_time":
            continue
        is_direction = "direction" in var
        hourly = []
        for hour in time_steps:
            idxs = hour_map.get(hour, [])
            if not idxs:
                hourly.append(None)
                continue
            vals = [values[i] for i in idxs if values[i] is not None]
            if not vals:
                hourly.append(None)
                continue
            if is_direction:
                hourly.append(circular_mean(vals))
            else:
                hourly.append(sum(vals) / len(vals))
        result[var] = hourly
    return result


def build_time_steps(start_dt, end_dt):
    from datetime import timedelta

    steps = []
    cur = start_dt
    while cur <= end_dt:
        steps.append(cur.strftime("%Y-%m-%dT%H:00:00Z"))
        cur += timedelta(hours=1)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RAWS data from Synoptic API")
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--start", required=True, help="YYYYmmddHHMM")
    parser.add_argument("--end", required=True, help="YYYYmmddHHMM")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--token", default=None, help="Synoptic token")
    parser.add_argument(
        "--network",
        default=None,
        help="Network IDs or short names (comma-separated). If omitted, uses RAWS networks.",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive stations (omit status=active).",
    )
    parser.add_argument(
        "--vars",
        default=None,
        help="Optional vars list for timeseries. If omitted, returns all available variables.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of station IDs per timeseries request.",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=168,
        help="Chunk size in hours for long time ranges (default 168 = 7 days).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests.",
    )
    args = parser.parse_args()

    token = get_token(args.token)
    minlon, minlat, maxlon, maxlat = args.bbox
    bbox_str = f"{minlon},{minlat},{maxlon},{maxlat}"

    if args.network:
        network = args.network
    else:
        raws_ids = get_raws_network_ids(token)
        if not raws_ids:
            raise SystemExit("Could not find RAWS networks via /networks. Pass --network.")
        network = ",".join(raws_ids)

    meta_params = {
        "token": token,
        "bbox": bbox_str,
        "network": network,
    }
    if not args.include_inactive:
        meta_params["status"] = "active"

    meta_url = f"{BASE_URL}stations/metadata?{urllib.parse.urlencode(meta_params)}"
    meta = http_get_json(meta_url)
    stations = meta.get("STATION", [])
    if not stations:
        raise SystemExit("No stations returned for bbox/network.")

    stids = [s.get("STID") for s in stations if s.get("STID")]
    stids = sorted(set(stids))

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Save station metadata
    stations_out = os.path.join(output_dir, "raws_stations.json")
    with open(stations_out, "w", encoding="utf-8") as f:
        json.dump({"STATION": stations, "SUMMARY": meta.get("SUMMARY", {})}, f, indent=2)

    # Build hourly time steps
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(args.start, "%Y%m%d%H%M")
    end_dt = datetime.strptime(args.end, "%Y%m%d%H%M")
    time_steps = build_time_steps(start_dt, end_dt)
    time_index = {t: i for i, t in enumerate(time_steps)}

    data_out = {
        "metadata": {
            "bbox": args.bbox,
            "network": network,
            "start": args.start,
            "end": args.end,
            "time_steps": time_steps,
        },
        "data": {},
    }

    # Initialize per-station containers
    for stid in stids:
        data_out["data"][stid] = {"observations": {}}

    # Chunk over time range to avoid huge responses
    chunk_hours = max(1, args.chunk_hours)
    cur_start = start_dt
    while cur_start <= end_dt:
        cur_end = min(end_dt, cur_start + timedelta(hours=chunk_hours - 1))
        chunk_steps = build_time_steps(cur_start, cur_end)
        chunk_start = cur_start.strftime("%Y%m%d%H%M")
        chunk_end = cur_end.strftime("%Y%m%d%H%M")

        for batch in chunks(stids, args.chunk_size):
            params = {
                "token": token,
                "stid": ",".join(batch),
                "start": chunk_start,
                "end": chunk_end,
                "obtimezone": "utc",
            }
            if args.vars:
                params["vars"] = args.vars
            ts_url = f"{BASE_URL}stations/timeseries?{urllib.parse.urlencode(params)}"
            ts = http_get_json(ts_url)
            station_list = ts.get("STATION", [])
            for station in station_list:
                stid = station.get("STID")
                if not stid:
                    continue
                if "latitude" not in data_out["data"][stid]:
                    data_out["data"][stid].update(
                        {
                            "latitude": station.get("LATITUDE"),
                            "longitude": station.get("LONGITUDE"),
                            "elevation": station.get("ELEVATION"),
                        }
                    )
                obs = station.get("OBSERVATIONS", {})
                hourly = aggregate_hourly(obs, chunk_steps)
                obs_store = data_out["data"][stid]["observations"]
                for var, values in hourly.items():
                    if var not in obs_store:
                        obs_store[var] = [None] * len(time_steps)
                    for i, hour in enumerate(chunk_steps):
                        val = values[i]
                        if val is None:
                            continue
                        idx = time_index[hour]
                        obs_store[var][idx] = val
            time.sleep(args.sleep)

        cur_start = cur_end + timedelta(hours=1)

    data_out_path = os.path.join(output_dir, "raws_timeseries_hourly.json")
    with open(data_out_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2)

    print(f"Saved stations: {stations_out}")
    print(f"Saved hourly data: {data_out_path}")


if __name__ == "__main__":
    main()
