# pages/settings_manager.py

import os
import json
import requests
import pathlib
import logging
from typing import Dict, Any
import requests
import time

# REST API endpoint (local Flask server on the Pi)
SETTINGS_URL = "http://127.0.0.1:8000/settings"

# Fallback file path (if HTTP fails)
LOCAL_BACKUP = pathlib.Path(__file__).with_name("settings_backup.json")

_DEFAULT_TIMEOUT = 2.0
_RETRIES = 3
_RETRY_DELAY = 0.2

logger = logging.getLogger(__name__)

def save_settings(data: Dict[str, Any]) -> None:
    for attempt in range(1,_RETRIES + 1):
        try:
            resp = requests.put(SETTINGS_URL, json = data, timeout = _DEFAULT_TIMEOUT)

            resp.raise_for_status()
            logger.info(" Settings saved to REST service (attempt %d).", attempt)
            return 

        except Exception as e:
            logger.warning("Post attempt %d failed: %s", attempt, e)
            time.sleep(_RETRY_DELAY)

    try:
        LOCAL_BACKUP.write_text(json.dumps(data, indent=2))
        logger.info(" Settings saved to local backup (%s)", LOCAL_BACKUP)
    except Exception as e:
        logger.error(" Failed to save local backup: %s", e)
    
def load_settings() -> Dict[str, Any]:
    for attempt in range(1,_RETRIES + 1):
        try:
            resp = requests.get(SETTINGS_URL, timeout = _DEFAULT_TIMEOUT)
            resp.raise_for_status()
            logger.info(" Settings loaded from REST (attempt %d).", attempt)
            try :
                data = resp.json() or {}
                return data

            except Exception as e:
                logger.warning("JSON decode from REST failed : %s",ex)
                break

        except Exception as e:
                logger.debug("GET attempt %d failed: %s",attempt, e)
                time.sleep(_RETRY_DELAY)

    if LOCAL_BACKUP.exists():
        try:
            txt = LOCAL_BACKUP.read_text()
            return json.loads(txt)
        except Exception as e:
            logger.error(" Faile to load loacal backup: %s", e)
            return {}
