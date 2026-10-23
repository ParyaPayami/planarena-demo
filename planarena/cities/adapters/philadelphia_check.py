"""Independent check of Philadelphia's digitised plan, and a sample sheet for checking it by eye.

For every district plan used, the map is registered again on its own pages, and each parcel's category is read in
two ways: from the vector shape that contains the parcel's point (the method used) and from the colour of the rendered
page around that point (pixels matched to the legend's rendered colours). Agreement between the two measures how
often the category drawn on a parcel is the one the method assigns, independently of the shape parsing (both use the
same registration, which the shifted-map control in the main run checks).

Also written: a random sample of parcels (10 per district) with page, page position and both readings, and a contact
sheet of the rendered map around each sampled point, for the author to check against the PDF by eye. No result of
that manual check is assumed here.

Run:  python -I -m planarena.cities.adapters.philadelphia_check
  -> results/validation/philadelphia_plan_check.json, results/validation/philadelphia_check/{sample.csv,sheet_*.png}
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pdfplumber
import shapely

from planarena.cities.adapters import philadelphia_plan as pp
from planarena.cities.registry import city_dir
from planarena.config import ROOT
from planarena.national.common import download
from planarena.sources import pdfplan
from planarena.sources.arcgis import fetch_layer

OUT = ROOT / "results" / "validation" / "philadelphia_plan_check.json"
SHEETS = ROOT / "results" / "validation" / "philadelphia_check"
PER_DISTRICT = 10


def _sheet(rows: list[dict], images: dict, path) -> None:
    from PIL import Image, ImageDraw

    size, cols = 160, 5
    sheet = Image.new("RGB", (cols * size, ((len(rows) + cols - 1) // cols) * (size + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(rows):
        img = images[r["page"]]
        cx, cy = int(r["px"]), int(r["py"])
        h = size // 2
        crop = np.ones((size, size, 3), np.float32)
        y0, x0 = max(cy - h, 0), max(cx - h, 0)
        part = img[y0:cy + h, x0:cx + h]
        crop[(y0 - (cy - h)):(y0 - (cy - h)) + part.shape[0], (x0 - (cx - h)):(x0 - (cx - h)) + part.shape[1]] = part
        tile = Image.fromarray((crop * 255).astype(np.uint8))
        d = ImageDraw.Draw(tile)
        d.line([(h - 8, h), (h + 8, h)], fill=(0, 0, 0), width=1)
        d.line([(h, h - 8), (h, h + 8)], fill=(0, 0, 0), width=1)
        X, Y = (n % cols) * size, (n // cols) * (size + 28)
        sheet.paste(tile, (X, Y))
        draw.text((X + 2, Y + size + 1), f"#{r['sample']} p.{r['page']}", fill=(0, 0, 0))
        draw.text((X + 2, Y + size + 13), str(r["vector_label"])[:26], fill=(0, 0, 0))
    sheet.save(path)


def main(seed: int = 0) -> dict:
    report = json.loads(pp.REPORT.read_text())
    lots = gpd.read_parquet(city_dir("4260000") / "parcels.geoparquet")
    twin = pd.read_parquet(city_dir("4260000") / "parcels.parquet", columns=["bbl"])
    stored = pd.read_parquet(pp.OUT)
    rp = lots.geometry.representative_point()
    dist = fetch_layer(pp.DISTRICTS_URL, pp.RAW / "planning_districts.parquet",
                       ["objectid", "dist_name", "abbrev"]).to_crs(lots.crs)
    by_file: dict[str, list[str]] = {}
    for name, fid in pp.PLANS.items():
        by_file.setdefault(fid, []).append(name)
    rng = np.random.default_rng(seed)
    res = {"test": __doc__.strip().split("\n\n")[1].replace("\n", " "), "districts": {}}
    SHEETS.mkdir(parents=True, exist_ok=True)
    sample = []
    for fid, names in by_file.items():
        key = " + ".join(names)
        if not report.get(key, {}).get("used"):
            continue
        pages = report[key]["pages"]
        pdf = download(pp.DRIVE.format(fid), pp.RAW / "plans" / f"{'_'.join(n.lower().replace(' ', '_') for n in names)}.pdf")
        n_pages = len(pdfplan.page_texts(pdf))
        area = shapely.union_all(list(dist[dist.dist_name.isin(names)].geometry))
        inside = rp.within(area).to_numpy()
        idx = np.nonzero(inside)[0]
        skip = set(range(1, n_pages + 1)) - set(pages)
        r = pdfplan.digitise(pdf, area, lots[inside], log=lambda s: None, skip_pages=skip)
        if r is None:
            res["districts"][key] = {"reregistered": False}
            continue
        vec, pix = r["labels"], r["pixel_labels"]
        both = (vec != "") & (pix != "")
        same_as_stored = float((vec == stored.plan_label.fillna("").to_numpy()[idx]).mean())
        conf = pd.crosstab(pd.Series(vec[both], name="vector"), pd.Series(pix[both], name="pixels"))
        off = conf.stack()
        off = off[[a != b for a, b in off.index]].sort_values(ascending=False).head(3)
        res["districts"][key] = {
            "pages": pages, "parcels": int(len(idx)), "vector_labelled": int((vec != "").sum()),
            "pixel_labelled": int((pix != "").sum()), "both": int(both.sum()),
            "agreement": round(float((vec[both] == pix[both]).mean()), 4) if both.any() else None,
            "same_as_stored_labels": round(same_as_stored, 4),
            "median_error_ft": round(r["transform"]["median_match_ft"], 2),
            "top_disagreements": [{"vector": a, "pixels": b, "parcels": int(v)} for (a, b), v in off.items()]}
        print(key, res["districts"][key], flush=True)
        # sample for the manual check
        tf = r["transform"]
        s_, R, t = tf["s"], np.array(tf["R"]), np.array(tf["t"])
        xy = np.c_[rp.x.to_numpy()[idx], rp.y.to_numpy()[idx]]
        src = (R.T @ ((xy - t) / s_).T).T
        pick = rng.choice(np.nonzero(vec != "")[0], size=min(PER_DISTRICT, int((vec != "").sum())), replace=False)
        images = {pg: pdfplan.render(pdf, pg) for pg in pages}
        with pdfplumber.open(str(pdf)) as doc:
            widths = [float(doc.pages[pg - 1].width) for pg in pages]
        offs = np.cumsum([0.0] + widths[:-1])
        rows = []
        for j in pick:
            x, y = src[j, 0], -src[j, 1]
            k = int(np.searchsorted(offs, x, side="right") - 1)
            if not 0 <= k < len(pages):
                continue
            rows.append({"sample": len(sample) + len(rows) + 1, "district": key, "bbl": str(twin.bbl.iloc[idx[j]]),
                         "page": pages[k], "page_x_pt": round(float(x - offs[k]), 1), "page_y_pt": round(float(y), 1),
                         "px": (x - offs[k]) * 300 / 72, "py": y * 300 / 72,
                         "vector_label": vec[j], "pixel_label": pix[j], "author_label": ""})
        _sheet(rows, images, SHEETS / f"sheet_{key.lower().replace(' + ', '_').replace(' ', '_')}.png")
        sample += rows
    pd.DataFrame(sample).drop(columns=["px", "py"]).to_csv(SHEETS / "sample.csv", index=False)
    d = [v for v in res["districts"].values() if v.get("both")]
    tot = sum(v["both"] for v in d)
    res["overall"] = {"districts": len(d), "parcels_read_both_ways": tot,
                      "agreement": round(sum(v["agreement"] * v["both"] for v in d) / tot, 4) if tot else None,
                      "agreement_range": [min(v["agreement"] for v in d), max(v["agreement"] for v in d)] if d else None,
                      "manual_sample": len(sample)}
    OUT.write_text(json.dumps(res, indent=1))
    print(res["overall"])
    return res


if __name__ == "__main__":
    main()
