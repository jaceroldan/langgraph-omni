import requests
import urllib.parse

VITE_LOCALHOST = "http://host.docker.internal:8000"


def fetch_task_counts(auth_token, workforce_id):
    headers = {
        "Authorization": auth_token
    }
    url = f"{VITE_LOCALHOST}/api-sileo/v4/hqzen/task-count/filter/?workforce_id={workforce_id}"
    response = requests.get(url, headers=headers)

    return response.json()


def fetch_shift_logs(auth_token, employment_id, shift_start):
    encoded_datetime = urllib.parse.quote(shift_start)
    headers = {
        "Authorization": auth_token
    }
    url = f"{VITE_LOCALHOST}/api-sileo/ai/timelogging/time-log/filter/?employment_id={employment_id}&shift_start={encoded_datetime}"  # noqa
    response = requests.get(url, headers=headers)

    return response.json()


def create_card(auth_token, data: dict):
    headers = {
        "Authorization": auth_token
    }

    user_id = data.get("user_id")
    title = data.get("title")
    is_public = data.get("is_public", True)

    url = f"{VITE_LOCALHOST}/api-sileo/v1/board/card-panel/create/"
    data = {
        "creator": user_id,
        "assignees": user_id,
        "title": title,
        "column": "213",
        "is_public": is_public
    }
    response = requests.post(url, data=data, headers=headers)
    return response.status_code
