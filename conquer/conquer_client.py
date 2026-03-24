"""
Conquer App API Client
======================
Fill in BASE_URL, TOKEN, and endpoint details after running intercept.py.

Usage:
  python conquer_client.py --list-games
  python conquer_client.py --signup GAME_ID
"""

import argparse
import json
import sys
import requests

# --- Fill these in after running analyze_traffic.py ---
BASE_URL = "https://api.conquerapp.io"   # update once discovered
TOKEN = ""                                # paste your auth token here
# ------------------------------------------------------

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "User-Agent": "Conquer/1.0 (Android; com.social_cludder.conquer_kotlin)",
}

session = requests.Session()
session.headers.update(HEADERS)


def login(email: str, password: str) -> str:
    """Login and return auth token."""
    resp = session.post(f"{BASE_URL}/auth/login", json={
        "email": email,
        "password": password,
    })
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("access_token") or data.get("auth_token")
    if not token:
        print("Login response:", json.dumps(data, indent=2))
        raise ValueError("Could not find token in login response")
    session.headers["Authorization"] = f"Bearer {token}"
    return token


def list_games(lat=40.7128, lon=-74.0060, radius_miles=10):
    """List available pickup games near a location."""
    resp = session.get(f"{BASE_URL}/games", params={
        "lat": lat,
        "lon": lon,
        "radius": radius_miles,
    })
    resp.raise_for_status()
    return resp.json()


def get_game(game_id: str):
    """Get details for a specific game."""
    resp = session.get(f"{BASE_URL}/games/{game_id}")
    resp.raise_for_status()
    return resp.json()


def signup_for_game(game_id: str):
    """Sign up for a game."""
    resp = session.post(f"{BASE_URL}/games/{game_id}/join")
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Conquer App Client")
    parser.add_argument("--login", nargs=2, metavar=("EMAIL", "PASSWORD"),
                        help="Login and print auth token")
    parser.add_argument("--list-games", action="store_true",
                        help="List available games")
    parser.add_argument("--game", metavar="GAME_ID",
                        help="Show details for a game")
    parser.add_argument("--signup", metavar="GAME_ID",
                        help="Sign up for a game")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="API base URL (override)")
    args = parser.parse_args()

    if args.base_url != BASE_URL:
        global BASE_URL
        BASE_URL = args.base_url

    if not TOKEN and not args.login:
        print("Warning: no TOKEN set. Set TOKEN in this file or use --login first.")

    if args.login:
        email, password = args.login
        token = login(email, password)
        print(f"Token: {token}")
        return

    if args.list_games:
        games = list_games()
        if isinstance(games, list):
            for g in games:
                gid = g.get("id") or g.get("_id") or "?"
                name = g.get("name") or g.get("title") or str(g)[:60]
                print(f"  {gid}: {name}")
        else:
            print(json.dumps(games, indent=2))
        return

    if args.game:
        game = get_game(args.game)
        print(json.dumps(game, indent=2))
        return

    if args.signup:
        result = signup_for_game(args.signup)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
