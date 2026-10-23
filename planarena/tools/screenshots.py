"""Screenshots of the Studio for the paper, scripted so they can be regenerated.

Starts the PlanArena server and a headless Chrome, drives the Studio through its store (window.__planarena), and
saves PNGs at twice screen resolution to docs/figures/:

  ui_impact.png     Manhattan: a plan that turns parking lots and vacant land in East Harlem into homes, with the
                    impact scoreboard
  ui_layers.png     Dallas: the adopted plan layer and the inspector of a vacant or parking lot the plan designates
                    for homes or mixed use (existing use, plan, zoning, what each new use would yield)
  ui_optimizer.png  Manhattan: the optimizer's recommended plan for a district
  ui_vote.png       Manhattan: the simulated stakeholders' vote on the candidate plans
and three crops that stay legible in print: ui_impact_crop.png (map and scoreboard), ui_inspector.png (map and
inspector) and ui_optimizer_vote.png (the optimizer and vote panels side by side).

Run:  python -I -m planarena.tools.screenshots      (needs Google Chrome)
"""
from __future__ import annotations

import asyncio
import base64
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

from planarena.config import ROOT

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "docs" / "figures"
W, H, SCALE = 1200, 820, 3  # a narrower viewport than a desktop screen, so the interface text stays legible in print
LEFT, RIGHT, TOP = 250, 410, 58  # Studio column widths and top bar height (web/src/styles.css, CSS px)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5).read()
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


class Page:
    """A minimal Chrome DevTools Protocol client for one page."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str, promise: bool = False):
        r = await self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": promise})
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expr[:80]}: {r['exceptionDetails'].get('text')} {r['result'].get('description', '')}")
        return r["result"].get("value")

    async def until(self, expr: str, timeout: float = 120, what: str = "") -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(f"Boolean({expr})"):
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(what or expr)

    async def goto(self, url: str) -> None:
        await self.send("Page.navigate", {"url": url})
        await self.until("window.__planarena && window.__planarena.getState().data && "
                         "window.__planarena.getState().data.n > 0", 300, f"city data at {url}")
        await self.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'skip')?.click()")

    async def settle(self, seconds: float = 6) -> None:
        await self.until("!window.__planarena.getState().evaluating", 120, "evaluation")
        await asyncio.sleep(seconds)  # map tiles and transitions

    async def shot(self, name: str) -> None:
        r = await self.send("Page.captureScreenshot", {"format": "png"})
        (OUT / name).write_bytes(base64.b64decode(r["data"]))
        print("written", (OUT / name).relative_to(ROOT), flush=True)

    async def click(self, text: str) -> None:
        ok = await self.js(f"(() => {{ const b = [...document.querySelectorAll('button')].find(b => "
                           f"b.textContent.includes({json.dumps(text)})); if (b) b.click(); return !!b; }})()")
        if not ok:
            raise RuntimeError(f"no button {text!r}")


S = "window.__planarena.getState()"


async def run(base: str, chrome_port: int) -> None:
    info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{chrome_port}/json").read())
    target = next(t for t in info if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
        p = Page(ws)
        await p.send("Page.enable")
        await p.send("Runtime.enable")
        await p.send("Emulation.setDeviceMetricsOverride",
                     {"width": W, "height": H, "deviceScaleFactor": SCALE, "mobile": False})

        # 1. Manhattan: parking lots and vacant land in East Harlem (district 111) become homes
        await p.goto(f"{base}/")
        await p.js(f"""(() => {{
            const s = {S}, d = s.data, k = s.useKeys, prk = k.indexOf('prk'), vac = k.indexOf('vac');
            const idx = [];
            for (let i = 0; i < d.n; i++) if (d.cd[i] === 111 && (d.baseUse[i] === prk || d.baseUse[i] === vac)) idx.push(i);
            const pick = idx.slice(0, 30);
            s.paint('res', pick);
            s.selectMany(pick, false);
            let x = 0, y = 0; for (const i of pick) {{ x += d.lon[i]; y += d.lat[i]; }}
            s.flyTo(x / pick.length, y / pick.length, 15.2);
            s.setTab('impact');
            return pick.length; }})()""")
        await p.settle(8)
        await p.shot("ui_impact.png")

        # 3. Manhattan: the optimizer's recommendation for the default district
        await p.js(f"{S}.reset(); {S}.clearSelection(); {S}.setTab('ai');")
        await asyncio.sleep(1)
        await p.click("Find a recommended plan")
        await p.until(f"{S}.preview", 300, "optimizer result")
        await p.js(f"""(() => {{ const s = {S}, d = s.data; let x = 0, y = 0, m = 0;
            for (let i = 0; i < d.n; i++) if (d.cd[i] === 103) {{ x += d.lon[i]; y += d.lat[i]; m++; }}
            s.flyTo(x / m, y / m, 14.6); }})()""")
        await p.settle(8)
        await p.shot("ui_optimizer.png")

        # 4. Manhattan: the simulated stakeholders' vote
        await p.js(f"{S}.setTab('agents');")
        await asyncio.sleep(1)
        await p.click("Hold the vote")
        await p.until(f"{S}.vote", 600, "vote result")
        await p.js(f"""(() => {{ const s = {S}, v = s.vote;
            s.setPreview({{ label: v.decision.title, changes: v.plan_changes[v.decision.plan] ?? {{}} }}); }})()""")
        await p.settle(6)
        await p.shot("ui_vote.png")

        # 2. Dallas: the adopted plan layer and the inspector of a parcel whose plan differs from its use
        await p.goto(f"{base}/?city=4819000")
        await p.js(f"{S}.setMapLayer('plan')", promise=True)
        await p.until(f"{S}.classLayers", 120, "class layers")
        # a parking lot or vacant parcel that the plan designates for homes or mixed use, near Deep Ellum
        await p.js(f"""(() => {{ const s = {S}, d = s.data, L = s.classLayers;
            let best = -1, bd = 1e9; const cx = -96.7835, cy = 32.7840;
            for (let i = 0; i < d.n; i++) {{
              if (![14, 15].includes(L.existing[i]) || ![2, 3].includes(L.plan[i])) continue;
              const dd = (d.lon[i] - cx) ** 2 + (d.lat[i] - cy) ** 2;
              if (dd < bd) {{ bd = dd; best = i; }} }}
            s.select(best, false);
            s.flyTo(d.lon[best], d.lat[best], 15);
            return best; }})()""")
        await p.settle(10)
        await p.shot("ui_layers.png")


def crops() -> None:
    """Figure crops that stay legible at print size, cut along the Studio's columns: the map and scoreboard of the
    impact view, the map with the parcel inspector, and the optimizer and vote panels side by side."""
    from PIL import Image

    k = SCALE
    im = Image.open(OUT / "ui_impact.png")
    im.crop((LEFT * k, 0, W * k, H * k)).save(OUT / "ui_impact_crop.png")
    im = Image.open(OUT / "ui_layers.png")
    im.crop((LEFT * k, TOP * k, (W - RIGHT) * k, H * k)).save(OUT / "ui_inspector.png")
    a, b = Image.open(OUT / "ui_optimizer.png"), Image.open(OUT / "ui_vote.png")
    box = ((W - RIGHT) * k, TOP * k, W * k, H * k)
    pa, pb = a.crop(box), b.crop(box)
    gap = 12 * k
    both = Image.new("RGB", (pa.width + pb.width + gap, pa.height), "white")
    both.paste(pa, (0, 0))
    both.paste(pb, (pa.width + gap, 0))
    both.save(OUT / "ui_optimizer_vote.png")
    print("written docs/figures/ui_impact_crop.png, ui_inspector.png, ui_optimizer_vote.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port, cport = _port(), _port()
    server = subprocess.Popen([sys.executable, "-I", "-m", "planarena.api", "--port", str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={cport}",
                                   f"--user-data-dir={profile}", f"--window-size={W},{H}", "--hide-scrollbars",
                                   "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-first-run",
                                   "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            _wait_http(f"http://127.0.0.1:{port}/api/meta")
            _wait_http(f"http://127.0.0.1:{cport}/json/version")
            asyncio.run(run(f"http://127.0.0.1:{port}", cport))
        finally:
            chrome.terminate()
            server.terminate()
            chrome.wait(timeout=30)  # let Chrome release its profile before the directory is removed
    crops()


if __name__ == "__main__":
    main()
