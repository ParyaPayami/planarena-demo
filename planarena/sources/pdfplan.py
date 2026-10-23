"""Turn an adopted land-use plan published only as a PDF map into a parcel layer (digitised, flagged as such).

Works on vector maps (each parcel drawn as a filled shape, as in plans exported from GIS):
1. Legend: small filled swatches with a label on the same line; their fill colours are the plan's categories.
2. Shapes: every filled shape on the map page(s) whose colour is a legend colour, as polygons in "spread"
   coordinates (facing pages side by side; y grows downward).
3. Georeferencing: a similarity transform (scale, rotation, translation; district maps are often turned). The map's
   footprint is first matched to the district parcels' footprint as images (rotation and scale searched, translation
   by FFT, scored by Dice overlap); this is refined by matching shape centroids to the nearest parcel centroids
   (iterative closest point, least squares). The median matching distance is reported.
   Only shapes that show on the page as poppler renders it are used: maps often draw a wider area and clip it to the
   district, or mask it, and the vector shapes carry neither. Only the largest map on the spread is used (insets
   redraw an area at another scale).
4. Parcels: each parcel's representative point, mapped onto the page, takes the colour of the smallest shape that
   contains it.
Nothing is guessed where the map has no shape: those parcels keep no plan class.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import shapely
import shapely.affinity


def _rgb(c) -> tuple | None:
    if c is None:
        return None
    c = tuple(float(x) for x in (c if isinstance(c, (list, tuple)) else (c,)))
    if len(c) == 1:
        return (c[0],) * 3
    if len(c) == 4:  # CMYK -> RGB
        k = 1 - c[3]
        return tuple(round((1 - x) * k, 3) for x in c[:3])
    return tuple(round(x, 3) for x in c[:3])


def _lab(c) -> np.ndarray:
    """sRGB (0-1) -> CIELAB, so colour distances follow what a reader distinguishes (delta E)."""
    c = np.asarray(c, dtype=float)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def legend(page) -> tuple[list[tuple], list[str], list]:
    """The legend's swatches (small filled squares stacked in one column) and the text to the right of each:
    colours, labels and bounding boxes (the swatches themselves are not map shapes)."""
    sw = [r for r in page.rects if r.get("fill") and 6 < r["width"] < 25 and 6 < r["height"] < 25
          and abs(r["width"] - r["height"]) < 4]
    if not sw:
        return [], [], []
    col_x = pd.Series([round(r["x0"]) for r in sw]).mode().iloc[0]  # the legend column
    sw = sorted([r for r in sw if abs(r["x0"] - col_x) < 3], key=lambda r: r["top"])
    chars = page.chars  # characters, not words: legend lines can overlap and interleave as words
    cols, labs, boxes = [], [], []
    for r in sw:
        yc = (r["top"] + r["bottom"]) / 2
        cs = sorted([c for c in chars if abs((c["top"] + c["bottom"]) / 2 - yc) < 3.5 and 0 < c["x0"] - r["x1"] < 140],
                    key=lambda c: c["x0"])
        text, last = "", None
        for c in cs:
            if last is not None and c["x0"] - last > 0.25 * (c["x1"] - c["x0"] + 1) + 1.5:
                text += " "
            text += c["text"]
            last = c["x1"]
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cols.append(_rgb(r.get("non_stroking_color")))
            labs.append(text)
            boxes.append((r["x0"] - 1, r["top"] - 1, r["x1"] + 1, r["bottom"] + 1))
    return cols, labs, boxes


def match_colours(legend_cols: list[tuple], legend_labs: list[str], map_counts: dict, max_cost: float = 45.0,
                  near: float = 8.0) -> dict:
    """Map colour -> legend label. The legend may be CMYK and the map RGB, so colours differ: the legend's distinct
    colours are matched one-to-one to the most frequent map colours (Hungarian assignment on delta E); a pair further
    apart than `max_cost` is no match, and a rarer map colour takes the label of a matched map colour within `near`.
    White is never a category."""
    from scipy.optimize import linear_sum_assignment

    uniq, ulab = [], []
    for c, l in zip(legend_cols, legend_labs):
        if c not in uniq:
            uniq.append(c)
            ulab.append(l)
    white = lambda c: min(c) > 0.97
    top = [c for c, _ in sorted(map_counts.items(), key=lambda kv: -kv[1]) if not white(c)][: max(3 * len(uniq), 20)]
    if not top:
        return {}
    cost = np.linalg.norm(_lab(uniq)[:, None, :] - _lab(top)[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    table = {top[j]: ulab[i] for i, j in zip(rows, cols) if cost[i, j] <= max_cost}
    matched = list(table)
    for c in map_counts:
        if c in table or white(c) or not matched:
            continue
        d = np.linalg.norm(_lab(matched) - _lab(c), axis=1)
        if d.min() <= near:
            table[c] = table[matched[int(d.argmin())]]
    return table


def shapes(pages, legend_cols: list[tuple], legend_labs: list[str], exclude: list | None = None):
    """Filled shapes of the legend's categories on facing pages, as (polygons, labels, colour table, fill colours) in
    spread coordinates."""
    objs, counts, x_off = [], {}, 0.0
    for k, page in enumerate(pages):
        for o in page.curves + page.rects:
            if not o.get("fill"):
                continue
            col = _rgb(o.get("non_stroking_color"))
            if col is None:
                continue
            if exclude and any(o["x0"] >= e[0] and o["x1"] <= e[2] and o["top"] >= e[1] and o["bottom"] <= e[3]
                               for e in exclude[k]):
                continue
            objs.append((x_off, o, col))
            counts[col] = counts.get(col, 0) + 1
        x_off += float(page.width)
    table = match_colours(legend_cols, legend_labs, counts)
    polys, labs, fills = [], [], []
    for x_off, o, col in objs:
        lab = table.get(col)
        if lab is None:
            continue
        pts = o.get("pts") or [(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]), (o["x0"], o["bottom"])]
        if len(pts) < 3:
            continue
        g = shapely.make_valid(shapely.Polygon([(x + x_off, y) for x, y in pts]))
        if g.is_empty or g.area <= 0:
            continue
        polys.append(g)
        labs.append(lab)
        fills.append(col)
    return (np.array(polys, dtype=object), np.array(labs), {str(c): l for c, l in table.items()},
            np.array(fills, dtype=float).reshape(-1, 3))


def render(pdf_path, page: int, dpi: int = 300) -> np.ndarray:
    """A page as poppler renders it (RGB in 0-1): what a reader sees, with clipping paths, masks and drawing order
    applied (the vector shapes carry none of these)."""
    import subprocess
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                        str(pdf_path), f"{d}/p"], check=True, capture_output=True)
        return np.asarray(Image.open(f"{d}/p.png").convert("RGB"), dtype=np.float32) / 255


def visible(polys, fills: np.ndarray, images: list, widths: list, dpi: int = 300,
            tol: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Whether each shape shows on the rendered page. Each fill colour's rendered colour is learned from the page (the
    most common non-white pixel at points inside its shapes; the renderer's colour conversion need not match the
    file's); a shape shows if a pixel next to a point inside it is within `tol` (delta E) of its colour's rendered
    colour and no nearer to another fill's or to the paper. Shapes clipped away, masked or drawn over are dropped.
    Returns the mask and the rendered colours (Lab) of the fills and of the paper."""
    uf, own = np.unique(np.round(fills, 2), axis=0, return_inverse=True)  # near-identical fills are one colour
    own = own.ravel()
    pts = shapely.point_on_surface(polys)
    x, y = shapely.get_x(pts), shapely.get_y(pts)
    offs = np.cumsum([0.0] + list(widths[:-1]))
    page = np.searchsorted(offs, x, side="right") - 1
    k = dpi / 72
    win = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    px = np.zeros((len(win), len(polys), 3), dtype=np.float32)
    for pi, img in enumerate(images):
        m = np.nonzero(page == pi)[0]
        cx, cy = ((x[m] - offs[pi]) * k).astype(int), (y[m] * k).astype(int)
        for w, (dx, dy) in enumerate(win):
            px[w, m] = img[np.clip(cy + dy, 0, img.shape[0] - 1), np.clip(cx + dx, 0, img.shape[1] - 1)]
    lab_px = _lab(px)
    paper = _lab(np.array([[1.0, 1.0, 1.0]]))
    rendered = np.zeros((len(uf), 3))
    for i in range(len(uf)):
        c = px[4, own == i]  # centre pixels
        c = c[np.linalg.norm(_lab(c)[:, None] - paper[None], axis=2).min(1) > 3] if len(c) else c
        rendered[i] = uf[i]
        if len(c):
            u, n = np.unique(np.round(c, 2), axis=0, return_counts=True)
            # a colour the fill's shapes mostly show, and one a renderer could make of the fill (CMYK conversions
            # shift colours by up to about 55 delta E), not text or line work drawn over a few of them
            if n.max() >= max(3, 0.2 * len(c)) and np.linalg.norm(_lab(u[n.argmax()]) - _lab(uf[i])) <= 60:
                rendered[i] = u[n.argmax()]
    C = np.vstack([_lab(rendered), paper])
    d = np.linalg.norm(lab_px[:, :, None, :] - C[None, None], axis=3)  # window x shape x colour
    mine = d[:, np.arange(len(polys)), own]
    return ((mine <= tol) & (mine <= d.min(2) + 1e-6)).any(0), (C[:-1], paper)


def map_mask(images: list, colours: tuple, exclude: list, dpi: int = 300, step: int = 2,
             tol: float = 8.0) -> tuple[np.ndarray, float]:
    """Where the rendered spread shows one of the map's fill colours rather than paper (legend swatches blanked;
    colours = (fills, paper) in Lab, from visible): the map as a reader
    sees it, for registering its footprint. Returns the mask (rows = page y, columns = spread x) and its pixel size
    in page units."""
    h = max(im.shape[0] for im in images)
    parts = []
    for im, ex in zip(images, exclude):
        im = im[::step, ::step].copy()
        for b in ex:
            x0, y0, x1, y1 = (int(v * dpi / 72 / step) for v in b)
            im[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1.0
        fills, paper = colours
        lab = _lab(im)
        d = np.linalg.norm(lab[:, :, None, :] - fills[None, None], axis=3).min(2)
        dp = np.linalg.norm(lab[:, :, None, :] - paper[None, None], axis=3).min(2)
        m = (d <= tol) & (d < dp)  # a fill colour, nearer to it than to the paper
        parts.append(np.pad(m, ((0, (h + step - 1) // step - m.shape[0]), (0, 0))))
    return np.hstack(parts), 72.0 * step / dpi


def main_frame(polys, mask: tuple, gutters=(), gap_pt: float = 20.0, gutter_pt: float = 36.0) -> tuple[np.ndarray, tuple]:
    """Keep the largest map on the spread: insets (an area redrawn at a larger scale) and locator maps are separate
    patches of map colour. Patches closer than `gap_pt` are one map, and so are patches facing each other across a
    gutter (x positions in `gutters`, page units) within `gutter_pt`: a map drawn across two pages stays whole.
    Returns which shapes lie in it and the mask restricted to it."""
    from scipy.ndimage import binary_dilation, label

    img, px = mask
    r = max(1, int(round(gap_pt / px / 2)))
    grown = binary_dilation(img, iterations=r)
    w = int(round(gutter_pt / px))
    for g in gutters:
        c = int(round(g / px))
        lo, hi = max(c - w, 0), min(c + w, img.shape[1])
        rows = grown[:, lo:c].any(1) & grown[:, c:hi].any(1)
        grown[rows, lo:hi] = True
    lab, n = label(grown)
    if n <= 1:
        return np.ones(len(polys), bool), mask
    pts = shapely.point_on_surface(polys)
    rows = np.clip((shapely.get_y(pts) / px).astype(int), 0, img.shape[0] - 1)
    cols = np.clip((shapely.get_x(pts) / px).astype(int), 0, img.shape[1] - 1)
    comp = lab[rows, cols]
    size = np.bincount(lab[img], minlength=n + 1)
    size[0] = 0
    keep = int(size.argmax())
    return comp == keep, (img & (lab == keep), px)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform dst ~ s * R @ src + t (no reflection)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1, d])
    R = u @ D @ vt
    s = float(np.trace(np.diag(sig) @ D) / a.var(0).sum())
    return s, R, md - s * R @ ms


def _raster(geoms, x0: float, y1: float, cell: float, shape: tuple) -> np.ndarray:
    """Binary raster of polygons (map coordinates, y up) on a grid with top-left corner (x0, y1)."""
    from PIL import Image, ImageDraw

    img = Image.new("1", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(img)
    for g in geoms:
        for part in getattr(g, "geoms", [g]):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            xy = np.asarray(part.exterior.coords)
            dr.polygon([((x - x0) / cell, (y1 - y) / cell) for x, y in xy], fill=1)
    return np.asarray(img, dtype=np.float32)


def coarse_register(polys, parcels, s0: float = 1.0, mask=None) -> tuple[float, float, np.ndarray, float]:
    """Best (scale, rotation, translation) of the map's footprint onto the parcels' footprint, in two stages: every
    3 degrees in +-90 at 100 ft cells, then every 0.5 degree around the best at 40 ft cells. Returns s, rotation
    (degrees), t, Dice. mask: (image, pixel size) from map_mask, the map's footprint as rendered (else the shapes)."""
    s, deg, _, _ = _search(polys, parcels, 100.0, s0, np.arange(-90, 90.5, 3.0), np.linspace(0.75, 1.3, 12), mask)
    return _search(polys, parcels, 40.0, s, np.arange(deg - 3, deg + 3.25, 0.5), np.linspace(0.94, 1.06, 9), mask)


def _search(polys, parcels, cell: float, s0: float, rotations, scales,
            mask=None) -> tuple[float, float, np.ndarray, float]:
    """The map is drawn once as an image on the page; for each rotation and scale it is warped onto the parcels'
    grid, and the translation maximising the overlap (FFT) is found; candidates are compared by Dice overlap."""
    from scipy.ndimage import affine_transform
    from scipy.signal import fftconvolve

    geoms = list(parcels.geometry) if hasattr(parcels, "geometry") else list(parcels)
    bx = shapely.bounds(shapely.union_all(geoms))
    pad = 0.3 * max(bx[2] - bx[0], bx[3] - bx[1])
    X0, Y0, X1, Y1 = bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad
    H, W = int((Y1 - Y0) / cell) + 1, int((X1 - X0) / cell) + 1
    target = _raster(geoms, X0, Y1, cell, (H, W))
    tsum = float(target.sum())
    # the map on the page, y down, at a page cell that matches `cell` at scale s0
    pb = shapely.bounds(shapely.union_all(list(polys)))
    pc = cell / s0 / 2  # page units per pixel (twice finer than the target)
    ph, pw = int((pb[3] - pb[1]) / pc) + 2, int((pb[2] - pb[0]) / pc) + 2
    if mask is None:
        page = _raster([shapely.affinity.scale(g, 1, -1, origin=(0, 0)) for g in polys], pb[0], -pb[1], pc, (ph, pw))
    else:  # resample the rendered footprint onto the same page grid
        from scipy.ndimage import map_coordinates

        img, px = mask
        rr, cc = np.mgrid[0:ph, 0:pw]
        page = map_coordinates(img.astype(np.float32), [(pb[1] + rr * pc) / px, (pb[0] + cc * pc) / px], order=1)
        page = (page > 0.5).astype(np.float32)
    best = (-np.inf, None)
    for deg in rotations:
        th = np.radians(deg)
        for f in scales:
            s = s0 * f
            # map = s * R(th) @ (qx, -qy) + t for a page point q; t first centres the page footprint on the canvas
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            pcen = np.array([(pb[0] + pb[2]) / 2, -(pb[1] + pb[3]) / 2])  # page centre, y up
            mcen = np.array([(X0 + X1) / 2, (Y0 + Y1) / 2])
            t = mcen - s * R @ pcen
            # target pixel (row r, col c) -> map (X0 + c*cell, Y1 - r*cell) -> page_up = R^T (map - t) / s
            # -> page pixel row = (-page_up_y - pb1)/pc, col = (page_up_x - pb0)/pc: linear in (r, c)
            Rt = R.T / s
            M = np.array([[0, cell], [-cell, 0]])  # d[mx,my]/d[r,c]
            base = Rt @ (np.array([X0, Y1]) - t)
            J = Rt @ M  # d page_up / d (r, c)
            A = np.array([-J[1] / pc, J[0] / pc])
            off = np.array([(-base[1] - pb[1]) / pc, (base[0] - pb[0]) / pc])
            mov = affine_transform(page, A, offset=off, output_shape=(H, W), order=1, cval=0.0)
            if mov.sum() <= 0:
                continue
            mov = (mov > 0.5).astype(np.float32)
            cc = fftconvolve(target, mov[::-1, ::-1], mode="same")  # overlap (cells) at every shift
            k = np.unravel_index(int(cc.argmax()), cc.shape)
            score = 2 * float(cc[k]) / (float(mov.sum()) + tsum)  # Dice: neither a shrunken nor a bloated map wins
            if score > best[0]:
                dr, dc = k[0] - H // 2, k[1] - W // 2
                best = (score, (s, float(deg), t + np.array([dc * cell, -dr * cell])))
    s, deg, t = best[1]
    return s, deg, t, best[0]


def georeference(polys, district, parcel_xy: np.ndarray, parcel_area: float | None = None, parcels=None,
                 mask=None) -> dict:
    """Similarity transform (scale, rotation, translation) from spread coordinates (y down) to map coordinates. With
    `parcels`, the footprints are first registered as images (coarse_register; against the parcels and, with a
    rendered `mask`, the district outline); otherwise starts are tried every
    3 degrees in +-45. Then iterative closest point from each parcel centroid to the nearest shape centroid (this
    direction ignores shapes that are not lots, such as streets), keeping the closest 70%. The reported errors are
    parcel-to-nearest-shape distances."""
    from scipy.spatial import cKDTree

    src = np.array([[p.centroid.x, -p.centroid.y] for p in polys])  # flip y: north up
    # first scale: the district's extent over the map's (shape areas overstate the map: hatching and overlaps)
    pb, db = shapely.bounds(shapely.union_all(list(polys))), district.bounds
    s0 = float(np.sqrt((db[2] - db[0]) / (pb[2] - pb[0]) * (db[3] - db[1]) / (pb[3] - pb[1])))

    def match(s, R, t):
        return cKDTree((s * (R @ src.T)).T + t).query(parcel_xy)

    def icp(s, R, t, iters):
        for _ in range(iters):
            d, j = match(s, R, t)
            keep = d <= np.percentile(d, 70)
            s, R, t = _umeyama(src[j[keep]], parcel_xy[keep])
        return s, R, t, float(np.median(match(s, R, t)[0]))

    if parcels is not None:  # raster registration first: grids of small lots trap point matching one lot off
        # targets: the parcels' footprint, and (with a rendered mask) the district's outline, since a map clipped to
        # its district covers areas the parcel layer can miss; the target whose fit lies closer to the lots wins
        best = None
        for target in [parcels] + ([[district]] if mask is not None else []):
            s, deg, t, overlap = coarse_register(polys, target, s0=s0, mask=mask)
            th = np.radians(deg)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            r = icp(s, R, t, 40)
            if best is None or r[3] < best[3]:
                best = r
        s, R, t, _ = best
    else:
        best = None
        for deg in range(-45, 46, 3):
            th = np.radians(deg)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            t = parcel_xy.mean(0) - s0 * R @ src.mean(0)
            r = icp(s0, R, t, 8)
            if best is None or r[3] < best[3]:
                best = r
        s, R, t, _ = icp(*best[:3], 40)
    d, _ = match(s, R, t)
    return {"s": float(s), "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), "R": R.tolist(),
            "t": t.tolist(), "median_match_ft": float(np.median(d)), "p90_match_ft": float(np.percentile(d, 90)),
            "n_shapes": int(len(polys))}


def classify(polys, labs, transform: dict, parcel_points: np.ndarray) -> np.ndarray:
    """Plan label of each parcel point (map coordinates), or '' where no shape covers it."""
    s, R, t = transform["s"], np.array(transform["R"]), np.array(transform["t"])
    src = (R.T @ ((parcel_points - t) / s).T).T  # back onto the page (y up)
    pts = shapely.points(src[:, 0], -src[:, 1])  # page coordinates are y down
    i, k = shapely.STRtree(polys).query(pts, predicate="within")
    area = shapely.area(polys)
    h = pd.DataFrame({"i": i, "k": k, "a": area[k]}).sort_values(["i", "a"]).drop_duplicates("i")
    out = np.full(len(parcel_points), "", dtype=object)
    out[h.i.to_numpy()] = labs[h.k.to_numpy()]
    return out


def classify_pixels(transform: dict, parcel_points: np.ndarray, images: list, widths: list, colours: np.ndarray,
                    colour_labels: np.ndarray, paper: np.ndarray, dpi: int = 300, tol: float = 10.0) -> np.ndarray:
    """Plan label of each parcel point read from the rendered page instead of the vector shapes: the legend category
    whose rendered colour (Lab) is nearest the pixels around the point (3 x 3, majority), if within `tol` delta E and
    nearer than the paper; '' on line work, text or paper. An independent check of `classify` (same registration)."""
    s, R, t = transform["s"], np.array(transform["R"]), np.array(transform["t"])
    src = (R.T @ ((parcel_points - t) / s).T).T
    x, y = src[:, 0], -src[:, 1]
    offs = np.cumsum([0.0] + list(widths[:-1]))
    page = np.searchsorted(offs, x, side="right") - 1
    k = dpi / 72
    C = np.vstack([colours, paper])
    votes = np.full((9, len(x)), -1)
    for pi, img in enumerate(images):
        m = np.nonzero(page == pi)[0]
        cx, cy = ((x[m] - offs[pi]) * k).astype(int), (y[m] * k).astype(int)
        inside = (cx >= 0) & (cy >= 0) & (cx < img.shape[1]) & (cy < img.shape[0])
        m, cx, cy = m[inside], cx[inside], cy[inside]
        for w, (dx, dy) in enumerate((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            px = img[np.clip(cy + dy, 0, img.shape[0] - 1), np.clip(cx + dx, 0, img.shape[1] - 1)]
            d = np.linalg.norm(_lab(px)[:, None, :] - C[None], axis=2)
            near = d.argmin(1)
            ok = (near < len(colours)) & (d[np.arange(len(m)), near] <= tol)
            votes[w, m[ok]] = near[ok]
    out = np.full(len(x), "", dtype=object)
    for i in range(len(x)):
        v = votes[:, i][votes[:, i] >= 0]
        if len(v):
            out[i] = colour_labels[np.bincount(v).argmax()]
    return out


def page_texts(pdf_path) -> dict[int, str]:
    """Each page's text, whitespace collapsed (poppler's pdftotext; fast on large layout files)."""
    import subprocess

    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout).group(1))
    return {i: re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf_path), "-"],
                                                  capture_output=True, text=True).stdout) for i in range(1, n + 1)}


def candidate_pages(pdf_path, phrase: str = "Proposed Land Use", texts: dict | None = None) -> list[int]:
    """1-based pages whose text contains the phrase."""
    texts = texts or page_texts(pdf_path)
    return [i for i, t in texts.items() if phrase.lower() in t.lower()]


EXISTING = "Existing Land Use"  # title of a plan's map of current land use (same legend as its plan map)


def digitise(pdf_path, district, parcels, phrases=("Proposed Land Use", "Future Land Use"), log=print,
             skip_pages=()) -> dict | None:
    """Plan label of every parcel of a district from a PDF plan's land-use map (see the module docstring).
    parcels: GeoDataFrame of the district's parcels (map CRS). Tries each page carrying one of the phrases, alone and
    with either facing page, with the legend read from whichever page of the spread has one; spreads with a page titled
    "Existing Land Use", or with a page in `skip_pages`, are skipped (that map shares the legend). Keeps the
    registration with the smallest median error. None if no map fits."""
    import pdfplumber

    rp = parcels.geometry.representative_point()
    xy = np.c_[rp.x.to_numpy(), rp.y.to_numpy()]
    best, images, legends, tried = None, {}, {}, set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = len(pdf.pages)
        texts = page_texts(pdf_path)
        pages_with = sorted({pg for ph in phrases for pg in candidate_pages(pdf_path, ph, texts)})
        for pg in pages_with:
            for spread in ([pg], [pg, pg + 1], [pg - 1, pg]):
                if min(spread) < 1 or max(spread) > n or tuple(spread) in tried:
                    continue
                tried.add(tuple(spread))
                if any(EXISTING in texts[i] or i in skip_pages for i in spread):  # today's land use, not the plan
                    continue
                for i in spread:
                    if i not in legends:
                        legends[i] = legend(pdf.pages[i - 1])
                lp = max(spread, key=lambda i: len(set(legends[i][1])))
                cols, labs, boxes = legends[lp]
                if len(set(labs)) < 6:
                    continue
                pages = [pdf.pages[i - 1] for i in spread]
                excl = [boxes if i == lp else [] for i in spread]
                polys, pl, table, fills = shapes(pages, cols, labs, exclude=excl)
                if len(polys) < 200:
                    continue
                imgs = [images[i] if i in images else images.setdefault(i, render(pdf_path, i)) for i in spread]
                seen, rendered = visible(polys, fills, imgs, [float(p.width) for p in pages])
                n_all = len(polys)
                # the legend category of each rendered fill colour (as in visible: near-identical fills are one)
                _, own = np.unique(np.round(fills, 2), axis=0, return_inverse=True)
                own = own.ravel()
                fill_labels = np.array([pd.Series(pl[own == i]).mode().iloc[0] for i in range(len(rendered[0]))],
                                       dtype=object)
                polys, pl = polys[seen], pl[seen]
                if len(polys) < 200:
                    continue
                mask = map_mask(imgs, rendered, excl)
                inside, mask = main_frame(polys, mask, gutters=np.cumsum([float(p.width) for p in pages])[:-1])
                polys, pl = polys[inside], pl[inside]
                if len(polys) < 200:
                    continue
                tf = georeference(polys, district, xy, parcels=parcels, mask=mask)
                lab = classify(polys, pl, tf, xy)
                cover = float((lab != "").mean())
                log(f"    pages {spread}: {len(polys):,} of {n_all:,} shapes visible, median {tf['median_match_ft']:.1f} ft, coverage {cover:.1%}")
                if cover >= 0.7 and (best is None or tf["median_match_ft"] < best["transform"]["median_match_ft"]):
                    shifted = {**tf, "t": [tf["t"][0] + 300.0, tf["t"][1]]}  # control: the map 300 ft off
                    best = {"pages": spread, "transform": tf, "coverage": cover, "labels": lab, "colours": table,
                            "control_labels": classify(polys, pl, shifted, xy),
                            "pixel_labels": classify_pixels(tf, xy, imgs, [float(p.width) for p in pages],
                                                            rendered[0], fill_labels, rendered[1]),
                            "shapes_visible": int(len(polys)), "shapes_drawn": int(n_all),
                            "legend": dict(zip([str(c) for c in cols], labs))}
    return best
