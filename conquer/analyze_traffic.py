"""
Analyze captured Conquer app traffic from conquer_traffic.json.

After running intercept.py and capturing traffic from the app, run:
  python analyze_traffic.py

This will print a summary of discovered endpoints and extract auth tokens.
"""

import json
import sys
from collections import defaultdict
from urllib.parse import urlparse


def load_traffic(path="conquer_traffic.json"):
    with open(path) as f:
        return json.load(f)


def analyze(traffic):
    endpoints = defaultdict(list)
    auth_tokens = set()
    base_urls = set()

    for entry in traffic:
        url = entry["url"]
        parsed = urlparse(url)
        base_urls.add(f"{parsed.scheme}://{parsed.netloc}")
        endpoint_key = f"{entry['method']} {parsed.path}"
        endpoints[endpoint_key].append(entry)

        # Look for auth tokens in headers
        headers = entry.get("request_headers", {})
        for key, val in headers.items():
            key_lower = key.lower()
            if key_lower in ("authorization", "x-auth-token", "x-api-key", "token"):
                auth_tokens.add(f"{key}: {val}")

        # Check response for tokens
        resp_body = entry.get("response_body")
        if isinstance(resp_body, dict):
            for field in ("token", "access_token", "auth_token", "jwt", "id_token"):
                if field in resp_body:
                    auth_tokens.add(f"response.{field}: {resp_body[field]}")

    print("=" * 60)
    print("BASE URLs DISCOVERED")
    print("=" * 60)
    for url in sorted(base_urls):
        print(f"  {url}")

    print()
    print("=" * 60)
    print("ENDPOINTS DISCOVERED")
    print("=" * 60)
    for endpoint, entries in sorted(endpoints.items()):
        print(f"\n  {endpoint}  ({len(entries)} calls)")
        # Show example request/response
        ex = entries[0]
        if ex.get("query"):
            print(f"    query params: {ex['query']}")
        if ex.get("request_body"):
            body = ex["request_body"]
            if isinstance(body, dict):
                print(f"    request fields: {list(body.keys())}")
        if isinstance(ex.get("response_body"), dict):
            print(f"    response fields: {list(ex['response_body'].keys())}")
        elif isinstance(ex.get("response_body"), list) and ex["response_body"]:
            print(f"    response: list[{len(ex['response_body'])}] of {type(ex['response_body'][0]).__name__}")

    print()
    print("=" * 60)
    print("AUTH TOKENS FOUND")
    print("=" * 60)
    if auth_tokens:
        for tok in auth_tokens:
            print(f"  {tok[:120]}")
    else:
        print("  (none found yet - try logging in first)")

    return base_urls, endpoints, auth_tokens


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "conquer_traffic.json"
    try:
        traffic = load_traffic(path)
        print(f"Loaded {len(traffic)} captured requests\n")
        analyze(traffic)
    except FileNotFoundError:
        print(f"No traffic file found at {path}")
        print("Run intercept.py first to capture app traffic.")
