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


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static dashboard files and JSON status without directory listing leaks."""

    def do_GET(self):
        if self.path == "/api/health":
            data_file = DASHBOARD_DIR / "demo_data.json"
            body = json.dumps({"status": "ok", "has_data": data_file.is_file()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

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
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
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
