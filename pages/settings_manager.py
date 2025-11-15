# pages/settings_manager.py

import os
import json
import requests
import pathlib
import logging

# REST API endpoint (local Flask server on the Pi)
SETTINGS_URL = "http://127.0.0.1:8000/settings"

# Fallback file path (if HTTP fails)
LOCAL_BACKUP = pathlib.Path(__file__).with_name("settings_backup.json")

def save_settings(data: dict):
    """Save settings via HTTP POST to REST API or fallback to local file."""
    try:
        response = requests.post(SETTINGS_URL, json=data, timeout=2)
        print("response:",response)
        response.raise_for_status()
        logging.info(" Settings saved to REST service.")
    except Exception as e:
        logging.warning("Failed to POST to REST → %s. Saving locally.", e)
        try:
            LOCAL_BACKUP.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logging.error(" Local backup failed: %s", e)

def load_settings() -> dict:
    """Load settings from REST API or fallback from local file."""
    try:
        response = requests.get(SETTINGS_URL, timeout=2)
        print("response1:",response)
        response.raise_for_status()
        logging.info("Loaded settings from REST.")
        return response.json() or {}
    
    except Exception as e:
        logging.warning("Failed to GET from REST → %s. Falling back to local file.", e)
        if LOCAL_BACKUP.exists():
            try:
                return json.loads(LOCAL_BACKUP.read_text())
            except Exception as e:
                logging.error(" Failed to load local backup: %s", e)
        return {}