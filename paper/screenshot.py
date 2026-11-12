#!/usr/bin/env python3
"""Capture paper/interface.png from a running search app (python app.py ... on localhost).

The app keeps the whole query state in the URL, so one page load reproduces a search.
Needs Playwright with a Chromium build:  pip install playwright && python -m playwright install chromium-headless-shell

    python paper/screenshot.py --url "http://127.0.0.1:5000/?q=...&model=dense&level=paragraph" --out paper/interface.png
"""

import argparse

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="paper/interface.png")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=2, help="device pixel ratio (2 = sharp in print)")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=args.scale, color_scheme="light")
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("#results article", timeout=60_000)  # results rendered
        page.wait_for_function("document.querySelectorAll('#results .katex').length > 0 || true")
        page.wait_for_timeout(500)  # let KaTeX finish laying out
        page.screenshot(path=args.out)
        browser.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
