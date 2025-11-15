# settings_server.py

from flask import Flask, request, jsonify
import json, os, threading
from pathlib import Path

# Save settings file in the same folder as this script
DATA_FILE = Path(__file__).parent / "settings.json"
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)  # Create dir if needed

_lock = threading.Lock()
app = Flask(__name__)

@app.route("/settings", methods=["GET", "POST"])
def handle_settings():
    with _lock:
        if request.method == "POST":
            with open(DATA_FILE, "w") as f:
                json.dump(request.get_json(force=True), f, indent=2)
            return "", 204
        elif request.method == "GET":
            if DATA_FILE.exists():
                with open(DATA_FILE) as f:
                    return jsonify(json.load(f))
            return jsonify({})

def run_flask_background():
    from werkzeug.serving import make_server
    server = make_server("127.0.0.1", 8000, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f" Flask settings server started on port 8000 → {DATA_FILE}")