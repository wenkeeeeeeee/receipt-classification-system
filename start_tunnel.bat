@echo off
REM start_tunnel.bat
REM ------------------
REM Starts a free Cloudflare Quick Tunnel pointing to your local Flask
REM app running on port 5000. This gives you a public URL anyone can
REM use, without ngrok's account requirement or aggressive timeout
REM behaviour on slow requests.
REM
REM Usage:
REM   1. Make sure app.py is already running in ANOTHER terminal window
REM      (python app.py) before running this script.
REM   2. Double-click this file, or run it from a terminal:
REM        start_tunnel.bat
REM   3. Look for a line like:
REM        https://random-words-here.trycloudflare.com
REM      That is your public demo URL. Share it with anyone.
REM
REM Notes:
REM   - This URL changes every time you restart the tunnel (same
REM     limitation as ngrok's free tier).
REM   - No account, signup, or authtoken is required for quick tunnels.
REM   - Keep this window open for as long as you want the public URL
REM     to keep working. Closing it ends the tunnel immediately.

echo ============================================================
echo   Starting Cloudflare Tunnel -^> http://localhost:5000
echo ============================================================
echo.
echo Make sure app.py is already running in another terminal first!
echo.

cloudflared.exe tunnel --url http://localhost:5000

pause
