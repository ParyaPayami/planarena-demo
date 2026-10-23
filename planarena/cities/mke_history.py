"""Annual snapshots of Milwaukee from the historical Master Property files, for redevelopment tests (2015-2026).

Each year's MPROP (assessment as of 1 January) is mapped into the parcel schema with today's parcel polygons, joined
by tax key; tax keys that no longer exist (split or merged parcels) have no polygon and are left out of that year.
The current MPROP (assessment year 2026) is the 2026 snapshot.

Floor area is not recorded consistently over time (only 15% of 2022 values equal 2021's, and 2016 lacks it for 27,186
parcels), so redevelopment tests in Milwaukee use new buildings (year built), which is stable.

Run:  python -I -m planarena.cities.mke_history
"""
from __future__ import annotations

import json

from planarena.config import DATA
from planarena.national.common import download

BASE = "https://data.milwaukee.gov/dataset/79e9d55a-c1ed-457c-ab63-cb8e36235ba8/resource/"
FILES = {
    2015: "fcd0b5eb-ee01-4c07-9af7-fef6ef243c00/download/mprop_2015.csv",
    2016: "5f67a8e0-7ce9-4b47-b72e-2a7fe0753f3f/download/mprop_2016.csv",
    2017: "b0dfa9e3-27d6-42f5-8614-52ac0485f7d5/download/mprop_2017.csv",
    2018: "61f12bd3-9348-4335-9f4d-b17b51282f90/download/mprop_2018.csv",
    2019: "6dcc1771-9867-45b6-9692-ed2da0f80fc7/download/mprop_2019.csv",
    2020: "324d7131-de71-4c9f-a1b8-ceea522104de/download/mprop_2020.csv",
    2021: "d1b65182-fff9-4400-9476-0aff30a6e0ea/download/mprop_2021.csv",
    2022: "a368cfd5-5504-4311-b394-df850da76367/download/parcel-tax-mprop-2022.csv",
    2023: "ed2f0086-9106-4177-b29d-b88fc346f461/download/parcel-tax-mprop-2023.csv",
    2024: "26700e70-5c86-461b-b75b-0e3ec73d9a57/download/parcel-tax-mprop-2024.csv",
    2025: "9e512b1d-2c64-448f-9e7a-5c2022d5948a/download/parcel-tax-mprop-2025.csv",
}
RAW = DATA / "raw" / "milwaukee_mprop_history"
HIST = DATA / "cities" / "5553000" / "history"
# 2016 is left out: 27,186 parcels have no floor area in that file although they have it in 2015 and 2017.
YEARS = [y for y in sorted(FILES) if y != 2016] + [2026]


def mprop_file(year: int):
    from planarena.cities.adapters.milwaukee import MPROP_URL, RAW as MRAW

    if year == 2026:
        return download(MPROP_URL, MRAW / "milwaukee_mprop" / "mprop.csv")
    return download(BASE + FILES[year], RAW / FILES[year].rsplit("/", 1)[1])


def build_all() -> dict:
    from planarena.cities.adapters import milwaukee

    out = {}
    for y in YEARS:
        d = HIST / str(y)
        if not (d / "parcels.parquet").exists():
            m = milwaukee.build(mprop_file(y), out=d, year=y)
            out[y] = {"parcels": m["n_parcels"], "missing_use": m["data_quality"]["missing_land_use"]}
            print(y, out[y], flush=True)
    return out


def snapshot_dir(year: int):
    return HIST / str(year)


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=1))
