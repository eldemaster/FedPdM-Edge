"""Lightweight zero-dependency web server for the FedPdM-Edge presentation dashboard."""
import argparse
import functools
import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
MAX_EDGE_UPDATE_BYTES = 64 * 1024
EDGE_UPDATE_FIELDS = {
    "asset_id",
    "cycle",
    "predicted_rul",
    "severity",
    "subsystem_shares_pct",
    "top_contributors",
    "top_drifting",
    "summary",
}
RAW_TELEMETRY_FIELDS = {"raw_sensors", "raw_window", "windows", "telemetry_raw"}


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static dashboard files and JSON status without directory listing leaks."""

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _live_assets(self):
        """Return server-local state, including for tests using a stock HTTPServer."""
        if not hasattr(self.server, "live_assets"):
            self.server.live_assets = {}
            self.server.live_assets_lock = threading.Lock()
        return self.server.live_assets, self.server.live_assets_lock

    @staticmethod
    def _contains_raw_telemetry(value):
        if isinstance(value, dict):
            if RAW_TELEMETRY_FIELDS.intersection(value):
                return True
            return any(DemoHandler._contains_raw_telemetry(item) for item in value.values())
        if isinstance(value, list):
            return any(DemoHandler._contains_raw_telemetry(item) for item in value)
        return False

    def do_GET(self):
        if self.path == "/api/health":
            data_file = DASHBOARD_DIR / "demo_data.json"
            self._send_json(200, {"status": "ok", "has_data": data_file.is_file()})
            return
        if self.path == "/api/live":
            assets, lock = self._live_assets()
            with lock:
                snapshot = dict(assets)
            self._send_json(200, {"mode": "live" if snapshot else "poc", "assets": snapshot})
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/edge_update":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length < 1:
            self._send_json(400, {"error": "JSON body required"})
            return
        if content_length > MAX_EDGE_UPDATE_BYTES:
            self._send_json(413, {"error": "payload exceeds 64 KiB"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "edge update must be a JSON object"})
            return
        if self._contains_raw_telemetry(payload):
            self._send_json(400, {"error": "raw telemetry is forbidden"})
            return
        unknown = set(payload).difference(EDGE_UPDATE_FIELDS)
        if unknown:
            self._send_json(400, {"error": f"unsupported fields: {', '.join(sorted(unknown))}"})
            return
        asset_id = payload.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip() or len(asset_id) > 256:
            self._send_json(400, {"error": "asset_id must be a non-empty string (max 256 characters)"})
            return

        assets, lock = self._live_assets()
        with lock:
            assets[asset_id] = payload
        self._send_json(200, {"status": "accepted", "asset_id": asset_id})

    def list_directory(self, path):
        """Never expose directory listings (e.g. __pycache__/); only index.html is served for "/"."""
        self.send_error(404, "Not Found")
        return None

    def log_message(self, format, *args):  # noqa: A002
        return


def main():
    parser = argparse.ArgumentParser(description="Launch FedPdM-Edge presentation dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(DemoHandler, directory=str(DASHBOARD_DIR))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        httpd.live_assets = {}
        httpd.live_assets_lock = threading.Lock()
        port = httpd.server_address[1]  # actual bound port (supports --port 0)
        url = f"http://{args.host}:{port}/"
        print(f"FedPdM-Edge Dashboard running at {url} (Ctrl+C to stop)", flush=True)
        if not args.no_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard.")


if __name__ == "__main__":
    main()
