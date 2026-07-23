#!/usr/bin/env python
"""Stage 7 live 3D frame viewer.

Run this on the cloud server alongside training to serve rendered
frames on port 6008 (already mapped in your autodl instance).

Usage:
    .venv/bin/python scripts/viz/frame_server.py /root/autodl-tmp/karbon_data/frames
"""

import argparse
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a frame directory over HTTP")
    ap.add_argument("frame_dir", type=str, help="Path to rendered frames directory")
    ap.add_argument("--port", type=int, default=6008)
    ap.add_argument("--interval", type=int, default=5,
                    help="Seconds between index rebuilds")
    args = ap.parse_args()

    target = Path(args.frame_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    index_html = target / "index.html"
    _rebuild_index(target, index_html)

    import threading

    def _periodic() -> None:
        while True:
            import time
            time.sleep(args.interval)
            try:
                _rebuild_index(target, index_html)
            except Exception:
                pass

    threading.Thread(target=_periodic, daemon=True).start()

    import os
    os.chdir(str(target))
    server = HTTPServer(("0.0.0.0", args.port), SimpleHTTPRequestHandler)
    print(f"[frame_server] serving {target} on :{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _rebuild_index(target: Path, index_path: Path) -> None:
    imgs = sorted(target.glob("*.png"), reverse=True)[:200]
    lines = [
        "<html><head><title>Stage 7 3D Agent View</title>",
        '<meta http-equiv="refresh" content="10">',
        "<style>body{font-family:monospace;background:#111;color:#0f0} "
        "img{border:1px solid #333;margin:4px} "
        "h2{color:#0f0}</style></head><body>",
        "<h2>Stage 7 · 3D Agent Live View</h2>",
        f"<p>{len(imgs)} frames (latest first)</p>",
    ]
    for p in imgs:
        name = p.name
        lines.append(f'<div><b>{name}</b><br><img src="{name}" width="320" loading="lazy"></div><hr>')
    lines.append("</body></html>")
    index_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
