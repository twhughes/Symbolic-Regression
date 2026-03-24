"""
Conquer Portal API Interceptor
================================
Uses Playwright to log into portal.conquerapp.io, intercepts all network
requests to discover the API base URL, endpoints, and auth tokens.

This works without a phone — runs entirely in a desktop browser.

Setup:
  pip install playwright
  playwright install chromium

Usage:
  python portal_intercept.py --email your@email.com --password yourpassword

Output: conquer_traffic.json (same format as intercept.py)
"""

import argparse
import json
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, Request, Response
except ImportError:
    print("Install playwright first:")
    print("  pip install playwright")
    print("  playwright install chromium")
    raise

PORTAL_URL = "https://portal.conquerapp.io/auth"
OUTPUT_FILE = "conquer_traffic.json"

captured = []


def is_api_request(url: str) -> bool:
    skip = [
        "google", "facebook", "analytics", "hotjar", "sentry",
        "fonts.gstatic", "amazonaws.com/static", ".css", ".png",
        ".jpg", ".svg", ".ico", ".woff",
    ]
    return not any(s in url for s in skip)


def run(email: str, password: str, headless: bool = False):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        request_bodies: dict[str, bytes] = {}

        def on_request(req: Request):
            if is_api_request(req.url):
                try:
                    request_bodies[req.url + req.method] = req.post_data_buffer or b""
                except Exception:
                    pass

        def on_response(resp: Response):
            if not is_api_request(resp.url):
                return
            try:
                body_bytes = resp.body()
            except Exception:
                body_bytes = b""

            entry = {
                "timestamp": time.time(),
                "method": resp.request.method,
                "url": resp.url,
                "status_code": resp.status,
                "request_headers": dict(resp.request.headers),
                "response_headers": dict(resp.headers),
                "request_body": None,
                "response_body": None,
            }

            # Request body
            raw_req = request_bodies.pop(resp.url + resp.request.method, b"")
            if raw_req:
                try:
                    entry["request_body"] = json.loads(raw_req)
                except Exception:
                    entry["request_body"] = raw_req.decode("utf-8", errors="replace")

            # Response body
            if body_bytes:
                try:
                    entry["response_body"] = json.loads(body_bytes)
                except Exception:
                    entry["response_body"] = body_bytes.decode("utf-8", errors="replace")

            captured.append(entry)
            print(f"  {resp.request.method} {resp.url[:80]} -> {resp.status}")

            # Save after each request
            with open(OUTPUT_FILE, "w") as f:
                json.dump(captured, f, indent=2, default=str)

        page.on("request", on_request)
        page.on("response", on_response)

        print(f"Opening {PORTAL_URL} ...")
        page.goto(PORTAL_URL, wait_until="networkidle")

        # Fill login form
        print("Filling login form...")
        try:
            page.fill('input[type="email"]', email)
            page.fill('input[type="password"]', password)
            page.click('button[type="submit"]')
            print("Submitted login, waiting for navigation...")
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception as e:
            print(f"Login form interaction failed: {e}")
            print("The page may look different — check the browser window.")
            if not headless:
                print("Please log in manually, then press Enter here...")
                input()

        print(f"\nLogged in (or attempted). Waiting 5s for any background requests...")
        page.wait_for_timeout(5000)

        print(f"\nSaved {len(captured)} requests to {OUTPUT_FILE}")
        print("Run: python analyze_traffic.py")

        browser.close()


def main():
    parser = argparse.ArgumentParser(description="Conquer portal traffic interceptor")
    parser.add_argument("--email", required=True, help="Conquer account email")
    parser.add_argument("--password", required=True, help="Conquer account password")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser headless (default: visible)")
    args = parser.parse_args()
    run(args.email, args.password, headless=args.headless)


if __name__ == "__main__":
    main()
