# Conquer Pickleball API Tools

Tools for interacting with the Conquer pickleball app.

## Step 1: Capture API Traffic

Since Conquer has no public API, you need to intercept the app's network traffic first.

### Install mitmproxy
```bash
pip install mitmproxy
```

### Start the proxy
```bash
cd conquer/
mitmproxy -s intercept.py --listen-port 8080
```

### Configure your phone
1. Connect phone and computer to the same WiFi
2. Set phone's proxy to `<your-computer-ip>:8080`
3. Visit `http://mitm.it` on phone and install the CA certificate
4. Open the Conquer app and:
   - Log in
   - Browse available games
   - Sign up for a game
   - View your profile

All traffic is saved to `conquer_traffic.json`.

## Step 2: Analyze the Traffic
```bash
python analyze_traffic.py
```

This prints:
- Base API URLs discovered
- All endpoints called
- Auth tokens captured

## Step 3: Use the Client

Edit `conquer_client.py` and fill in:
- `BASE_URL` — from the analysis output
- `TOKEN` — from the analysis output (or use `--login`)

Then:
```bash
# Login
python conquer_client.py --login your@email.com yourpassword

# List games
python conquer_client.py --list-games

# Sign up for a game
python conquer_client.py --signup GAME_ID
```

## Notes
- Tokens likely expire; re-run `--login` if you get 401 errors
- Endpoint names above (`/games`, `/games/{id}/join`) are guesses and will be updated after traffic capture
