#!/usr/bin/env python3
"""Refresh South America soybean weather data from World Ag Weather.

Outputs:
- sa_soy_weather_data.json
- sa_soy_weather_data.js
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

ROOT = Path(__file__).resolve().parent
TMP = Path("/tmp")


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_to_file(url: str, path: Path) -> None:
    path.write_bytes(fetch_bytes(url))


def tsv_rows(path: Path, psm: int = 11) -> List[Dict[str, object]]:
    out = subprocess.check_output(
        ["tesseract", str(path), "stdout", "--psm", str(psm), "tsv"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    rows: List[Dict[str, object]] = []
    for line in out.splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 12:
            continue
        rows.append(
            {
                "left": int(p[6]),
                "top": int(p[7]),
                "width": int(p[8]),
                "height": int(p[9]),
                "conf": float(p[10]) if p[10] != "-1" else -1.0,
                "text": p[11].strip(),
            }
        )
    return rows


def linear_fit(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    ys = np.array([p[0] for p in points], dtype=float)
    vs = np.array([p[1] for p in points], dtype=float)
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, vs, rcond=None)[0]
    return float(a), float(b)


def parse_summary(summary_html: str) -> Tuple[dt.date, Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    m = re.search(r"through\s*<b>(\d{1,2}\s+\w+\s+\d{4})</b>", summary_html, re.I)
    summary_date = dt.datetime.strptime(m.group(1), "%d %b %Y").date() if m else dt.date.today()

    soup = BeautifulSoup(summary_html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) == 5 and tds[0].get_text(strip=True):
            name = tds[0].get_text(" ", strip=True)
            vals = [float(td.get_text(strip=True).replace("+", "")) for td in tds[1:]]
            rows.append(
                {
                    "name": name,
                    "pcp60": vals[0],
                    "pcp180": vals[1],
                    "tmp60": vals[2],
                    "tmp180": vals[3],
                }
            )

    lookup = {r["name"]: r for r in rows}
    national = {
        "brazil": lookup["Brazil"],
        "argentina": lookup["Argentina"],
    }
    states = {
        "brazil": [
            lookup["Mato Grosso"],
            lookup["Parana"],
            lookup["Rio Grande do Sul"],
            lookup["Goias"],
            lookup["Mato Grosso do Sul"],
        ],
        "argentina": [
            lookup["Buenos Aires"],
            lookup["Cordoba"],
            lookup["Santa Fe"],
            lookup["Entre Rios"],
            lookup["Santiago del Estero"],
        ],
    }
    return summary_date, national, states


def extract_country_series(country: str, imgnum: str) -> Dict[str, object]:
    pcp_img = np.array(Image.open(TMP / f"fcstpcp_soybeans_{country}_{imgnum}.png").convert("RGB"))
    tmp_img = np.array(Image.open(TMP / f"fcsttmp_soybeans_{country}_{imgnum}.png").convert("RGB"))

    x0, x1 = 53, 604
    y0, y1 = 345, 458

    r = pcp_img[y0 : y1 + 1, x0 : x1 + 1, 0]
    g = pcp_img[y0 : y1 + 1, x0 : x1 + 1, 1]
    b = pcp_img[y0 : y1 + 1, x0 : x1 + 1, 2]
    green = (g > 180) & (r < 170) & (b < 80)

    seg: List[Tuple[int, int]] = []
    col = green.sum(axis=0)
    on = False
    start = 0
    for i, v in enumerate(col):
        if v > 0 and not on:
            start = i
            on = True
        if on and v == 0:
            seg.append((start, i - 1))
            on = False
    if on:
        seg.append((start, len(col) - 1))

    pcp_axis = Image.fromarray(pcp_img).convert("L").crop((0, 335, 90, 470))
    pcp_axis = ImageEnhance.Contrast(pcp_axis).enhance(3.0)
    pcp_axis = pcp_axis.point(lambda p: 255 if p > 170 else 0).resize((90 * 8, (470 - 335) * 8))
    pcp_axis_path = TMP / f"{country}_pcp_axis_extract.png"
    pcp_axis.save(pcp_axis_path)

    axis_points: List[Tuple[float, float]] = [(457.0, 0.0)]
    for row in tsv_rows(pcp_axis_path, 11):
        if float(row["conf"]) < 40:
            continue
        m = re.search(r"(\d\.\d+)", str(row["text"]))
        if not m:
            continue
        val = float(m.group(1))
        y = (float(row["top"]) + float(row["height"]) / 2) / 8 + 335
        axis_points.append((y, val))

    a_pcp, b_pcp = linear_fit(axis_points)

    daily_precip: List[float] = []
    centers: List[float] = []
    for s, e in seg:
        ys = np.where(green[:, s : e + 1])[0]
        if len(ys) == 0:
            continue
        y = 345 + int(ys.min())
        daily_precip.append(round(max(0.0, a_pcp * y + b_pcp), 2))
        centers.append(x0 + (s + e) / 2)

    tmp_axis = Image.fromarray(tmp_img).convert("L").crop((0, 95, 90, 460))
    tmp_axis = ImageEnhance.Contrast(tmp_axis).enhance(3.0)
    tmp_axis = tmp_axis.point(lambda p: 255 if p > 170 else 0).resize((90 * 6, (460 - 95) * 6))
    tmp_axis_path = TMP / f"{country}_tmp_axis_extract.png"
    tmp_axis.save(tmp_axis_path)

    tpts: List[Tuple[float, float]] = []
    for row in tsv_rows(tmp_axis_path, 11):
        if float(row["conf"]) < 40:
            continue
        txt = str(row["text"])
        if not re.fullmatch(r"\d{2}", txt):
            continue
        v = int(txt)
        if not (55 <= v <= 95):
            continue
        y = (float(row["top"]) + float(row["height"]) / 2) / 6 + 95
        tpts.append((y, float(v)))

    dedup: Dict[int, List[float]] = {}
    for y, v in tpts:
        dedup.setdefault(int(v), []).append(y)
    fit_pts = [(float(np.median(dedup[v])), float(v)) for v in sorted(dedup.keys())]
    a_tmp, b_tmp = linear_fit(fit_pts)

    daily_temp: List[float] = []
    y_lo, y_hi = (140, 440) if country == "argentina" else (180, 360)
    for cx in centers:
        xi = int(round(cx))
        cand: List[int] = []
        for dx in range(-4, 5):
            xx = min(max(0, xi + dx), tmp_img.shape[1] - 1)
            strip = tmp_img[y_lo:y_hi, xx : xx + 1, :]
            black = (strip[:, :, 0] < 60) & (strip[:, :, 1] < 60) & (strip[:, :, 2] < 60)
            ys = np.where(black)[0]
            if len(ys):
                cand.extend((y_lo + ys).tolist())
        if not cand:
            daily_temp.append(float("nan"))
        else:
            y = float(np.median(cand))
            daily_temp.append(round(a_tmp * y + b_tmp, 1))

    for i, v in enumerate(daily_temp):
        if not np.isnan(v):
            continue
        left = next((daily_temp[j] for j in range(i - 1, -1, -1) if not np.isnan(daily_temp[j])), None)
        right = next((daily_temp[j] for j in range(i + 1, len(daily_temp)) if not np.isnan(daily_temp[j])), None)
        if left is not None and right is not None:
            daily_temp[i] = round((left + right) / 2, 1)
        elif left is not None:
            daily_temp[i] = left
        elif right is not None:
            daily_temp[i] = right
        else:
            daily_temp[i] = 0.0

    while len(daily_precip) < 16:
        daily_precip.append(daily_precip[-1] if daily_precip else 0.0)
    while len(daily_temp) < 16:
        daily_temp.append(daily_temp[-1] if daily_temp else 0.0)

    daily_precip = daily_precip[:16]
    daily_temp = daily_temp[:16]

    cum: List[float] = []
    total = 0.0
    for v in daily_precip:
        total += v
        cum.append(round(total, 2))

    return {
        "daily_precip_in": daily_precip,
        "cum_precip_in": cum,
        "daily_temp_f": daily_temp,
        "total_precip_in": round(sum(daily_precip), 2),
        "avg_temp_f": round(sum(daily_temp) / len(daily_temp), 1),
    }


def main() -> None:
    imgnum = fetch_bytes("https://www.worldagweather.com/cgi-bin/ag/getcropimglabs.pl").decode("utf-8").split("|")[0].strip()

    summary_html = fetch_bytes("https://www.worldagweather.com/cgi-bin/ag/loadsummary.pl?crop=soybeans").decode("utf-8", "ignore")
    (TMP / "worldagweather_soy_summary.html").write_text(summary_html, encoding="utf-8")

    for country in ["brazil", "argentina"]:
        fetch_to_file(
            f"https://www.worldagweather.com/crops/fcstwx/fcstpcp_soybeans_{country}_{imgnum}.png",
            TMP / f"fcstpcp_soybeans_{country}_{imgnum}.png",
        )
        fetch_to_file(
            f"https://www.worldagweather.com/crops/fcstwx/fcsttmp_soybeans_{country}_{imgnum}.png",
            TMP / f"fcsttmp_soybeans_{country}_{imgnum}.png",
        )

    summary_date, national, states = parse_summary(summary_html)
    start = summary_date + dt.timedelta(days=1)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(16)]

    out = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_through": summary_date.isoformat(),
        "forecast_start": start.isoformat(),
        "image_number": imgnum,
        "dates": dates,
        "labels": [d[5:] for d in dates],
        "forecast": {
            "brazil": extract_country_series("brazil", imgnum),
            "argentina": extract_country_series("argentina", imgnum),
        },
        "anomaly_national": national,
        "anomaly_states": states,
    }

    json_path = ROOT / "sa_soy_weather_data.json"
    js_path = ROOT / "sa_soy_weather_data.js"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    js_path.write_text("window.SA_SOY_WEATHER_DATA = " + json.dumps(out, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")

    print(f"Refreshed data with image_number={imgnum}")
    print(f"Summary through: {out['summary_through']} | Forecast start: {out['forecast_start']}")
    print(f"Brazil total precip: {out['forecast']['brazil']['total_precip_in']} in")
    print(f"Argentina total precip: {out['forecast']['argentina']['total_precip_in']} in")


if __name__ == "__main__":
    main()
