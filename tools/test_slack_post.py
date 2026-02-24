import requests
import sys

WEBHOOK_URL = "https://hooks.slack.com/services/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"

payload = {"text": "[numbers3] test notification from local Streamlit app (automated test)"}

try:
    resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
    print("HTTP_STATUS:", resp.status_code)
    print("RESPONSE_BODY:", resp.text)
    if not (200 <= resp.status_code < 300):
        sys.exit(2)
except Exception as e:
    print("EXCEPTION:", str(e))
    sys.exit(3)
