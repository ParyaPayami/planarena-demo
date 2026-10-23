"""Download every open dataset the validation uses, and record where it came from.

Run with:  python -I -m planarena.tools.fetch_open_data         (skips files already present)
           python -I -m planarena.tools.fetch_open_data --check (hash only; no downloads)

Each file goes into its own folder under data/raw/ with a SOURCE.json (URL, query, SHA-256, retrieval date).
results/data_manifest.json lists the SHA-256 of every input so a reproduction can be compared file by file.
Socrata (NYC Open Data) extracts are live tables; a later download can differ from the one used for the results,
which the manifest makes visible.
MapPLUTO archives are fetched by planarena.twin.fetch (current release) and planarena.models.history (annual).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from planarena.config import DATA, ROOT

RAW = DATA / "raw"
NYC = "https://data.cityofnewyork.us/resource/"
LODES = "https://lehd.ces.census.gov/data/lodes/LODES8/ny/"
EIS = "https://www.nyc.gov/assets/planning/download/pdf/applicants/env-review/"
LL84_COLS = ("nyc_borough_block_and_lot,primary_property_type,property_gfa_calculated,site_eui_kbtu_ft,natural_gas_use_kbtu,"
             "electricity_use_grid_purchase,district_steam_use_kbtu,fuel_oil_2_use_kbtu,fuel_oil_4_use_kbtu")

# folder, file, url, Socrata query (or None), what it is used for
SOURCES = [
    ("lodes8_ny_2015", "ny_wac_S000_JT00_2015.csv.gz", LODES + "wac/ny_wac_S000_JT00_2015.csv.gz", None,
     "LEHD LODES 8 workplace jobs by block, 2015 (jobs-rate forward test, change prediction)"),
    ("lodes8_ny_2023", "ny_wac_S000_JT00_2023.csv.gz", LODES + "wac/ny_wac_S000_JT00_2023.csv.gz", None,
     "LEHD LODES 8 workplace jobs by block, 2023 (jobs calibration and validation)"),
    ("lodes8_ny_xwalk", "ny_xwalk.csv.gz", LODES + "ny_xwalk.csv.gz", None, "LODES block-to-tract crosswalk"),
    ("ll84_2015", "ll84_manhattan_2015_bbl.json", NYC + "77q4-nkfh.json",
     {"$select": "nyc_borough_block_and_lot,primary_property_type_self,property_gfa_self_reported,natural_gas_use_kbtu,"
                 "electricity_use_grid_purchase,electricity_use_grid_purchase_1,district_steam_use_kbtu,fuel_oil_2_use_kbtu,"
                 "fuel_oil_4_use_kbtu,site_eui_kbtu_ft,site_energy_use_kbtu",
      "$where": "upper(borough)='MANHATTAN'", "$limit": "50000"},
     "NYC LL84 benchmarking, calendar 2015 (change prediction of emissions)"),
    ("ll84_2023", "ll84_manhattan_2023_bbl.json", NYC + "5zyy-y8am.json",
     {"$select": LL84_COLS + ",report_year", "$where": "upper(borough)='MANHATTAN' AND report_year='2023'", "$limit": "50000"},
     "NYC LL84 benchmarking, calendar 2023 (emissions validation)"),
    ("ll84_2023_citywide", "ll84_nyc_2023_bbl.json", NYC + "5zyy-y8am.json",
     {"$select": LL84_COLS.replace("nyc_borough_block_and_lot,", "nyc_borough_block_and_lot,borough,") + ",report_year",
      "$where": "report_year='2023'", "$limit": "100000"},
     "NYC LL84 benchmarking, calendar 2023, all boroughs (choice of coefficients for the citywide twin)"),
    ("dob_co", "dob_co_legacy_mn.json", NYC + "bs8b-p36w.json",
     {"$select": "bbl,job_number,job_type,issue_type,c_o_issue_date",
      "$where": "upper(borough)='MANHATTAN' AND job_type in ('NB','A1') AND c_o_issue_date >= '2014-01-01'", "$limit": "200000"},
     "DOB certificates of occupancy, BIS (label audit)"),
    ("dob_co", "dob_co_now_mn.json", NYC + "pkdm-hqz6.json",
     {"$select": "bbl,application_number,job_type,c_of_o_filing_type,c_of_o_issuance_date,number_of_dwelling_units",
      "$where": "upper(borough)='MANHATTAN'", "$limit": "200000"},
     "DOB certificates of occupancy, DOB NOW (label audit)"),
    ("eis_east_harlem", "appenda_feis.pdf", EIS + "east-harlem/appenda_feis.pdf", None,
     "East Harlem Rezoning FEIS, Appendix A (development sites)"),
    ("eis_east_harlem", "01_feis.pdf", EIS + "east-harlem/01_feis.pdf", None, "East Harlem Rezoning FEIS, Chapter 1"),
    ("eis_soho_noho", "appenda-feis.pdf", EIS + "soho-noho/appenda-feis.pdf", None,
     "SoHo/NoHo Neighborhood Plan FEIS, Appendix A (development sites)"),
    ("eis_soho_noho", "01-feis.pdf", EIS + "soho-noho/01-feis.pdf", None, "SoHo/NoHo Neighborhood Plan FEIS, Chapter 1"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, query: dict | None, out: Path) -> None:
    cmd = ["curl", "-sfL", "-m", "900", "-A", "Mozilla/5.0", "-o", str(out)]
    if query:
        cmd += ["-G", url] + sum((["--data-urlencode", f"{k}={v}"] for k, v in query.items()), [])
    else:
        cmd.append(url)
    subprocess.run(cmd, check=True)


def fetch(check: bool = False) -> dict:
    manifest = {}
    for folder, name, url, query, what in SOURCES:
        d = RAW / folder
        f = d / name
        if not f.exists():
            if check:
                manifest[f"{folder}/{name}"] = None
                continue
            d.mkdir(parents=True, exist_ok=True)
            _get(url, query, f)
        src = d / "SOURCE.json"
        meta = json.loads(src.read_text()) if src.exists() else {}
        h = sha256(f)
        if meta.get(name, {}).get("sha256") != h:
            meta[name] = {"url": url, "query": query, "used_for": what, "sha256": h, "bytes": f.stat().st_size,
                          "recorded": dt.date.today().isoformat()}
            src.write_text(json.dumps(meta, indent=2))
        manifest[f"{folder}/{name}"] = h
    for p in sorted(RAW.glob("pluto_*/SOURCE.json")):
        s = json.loads(p.read_text())
        manifest[f"{p.parent.name}/{s.get('extract', '')}"] = s.get("extract_sha256")
        manifest[f"{p.parent.name}/archive"] = s.get("archive_sha256")
    for p in sorted((DATA / "reference").glob("*.json")):
        manifest[f"reference/{p.name}"] = sha256(p)
    if not check:
        (ROOT / "results" / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = fetch(check="--check" in sys.argv)
    missing = [k for k, v in m.items() if v is None]
    print(f"{len(m) - len(missing)} files recorded" + (f"; missing: {missing}" if missing else ""))
