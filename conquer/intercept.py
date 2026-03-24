"""
Conquer App API Interceptor
===========================
Run this as a mitmproxy addon to capture and log all Conquer app API traffic.

Setup:
  pip install mitmproxy
  mitmproxy -s intercept.py --listen-port 8080

On your phone:
  1. Set proxy to <your-ip>:8080
  2. Install mitmproxy CA cert: http://mitm.it
  3. Open the Conquer app and browse/sign up for games

Captured traffic is saved to conquer_traffic.json
"""

import json
import os
import time
from mitmproxy import http

CONQUER_DOMAINS = [
    "conquerapp.io",
    "conquerpickleballnyc.com",
    "social-cludder",
    "api.conquer",
]

OUTPUT_FILE = "conquer_traffic.json"
captured = []


def is_conquer_request(flow: http.HTTPFlow) -> bool:
    host = flow.request.pretty_host.lower()
    return any(domain in host for domain in CONQUER_DOMAINS)


def response(flow: http.HTTPFlow):
    if not is_conquer_request(flow):
        return

    req = flow.request
    resp = flow.response

    entry = {
        "timestamp": time.time(),
        "method": req.method,
        "url": req.pretty_url,
        "host": req.pretty_host,
        "path": req.path,
        "query": dict(req.query),
        "request_headers": dict(req.headers),
        "request_body": None,
        "status_code": resp.status_code,
        "response_headers": dict(resp.headers),
        "response_body": None,
    }

    # Capture request body
    if req.content:
        try:
            entry["request_body"] = json.loads(req.content)
        except Exception:
            entry["request_body"] = req.content.decode("utf-8", errors="replace")

    # Capture response body
    if resp.content:
        try:
            entry["response_body"] = json.loads(resp.content)
        except Exception:
            entry["response_body"] = resp.content.decode("utf-8", errors="replace")

    captured.append(entry)

    # Print summary to console
    print(f"[CONQUER] {req.method} {req.path} -> {resp.status_code}")
    if entry["request_body"]:
        print(f"  REQ: {json.dumps(entry['request_body'])[:200]}")
    if entry["response_body"]:
        body_preview = str(entry["response_body"])[:300]
        print(f"  RES: {body_preview}")
    print()

    # Save to file after each capture
    with open(OUTPUT_FILE, "w") as f:
        json.dump(captured, f, indent=2, default=str)
    print(f"  [saved to {OUTPUT_FILE}]")
