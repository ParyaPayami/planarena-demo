"""Run the PlanArena server: python -m planarena.api [--port 8787]"""
import argparse

import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8787)
a = ap.parse_args()
uvicorn.run("planarena.api.app:app", host=a.host, port=a.port)
