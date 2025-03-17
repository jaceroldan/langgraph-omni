import requests
import urllib.parse
import json

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
    url = f"{VITE_LOCALHOST}/api-sileo/ai/timelogging/time-log/filter/?employment_id={employment_id}&shift_start={encoded_datetime}"
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


def fetch_weekly_task_estimates(auth_token, employment_id, user_profile_pk, x_timezone):
    headers = {
        "Authorization": auth_token,
        "X-Timezone": x_timezone
    }
    url = f"{VITE_LOCALHOST}/api-sileo/v4/hqzen/task-assignments/filter/"
    url2 = f"{VITE_LOCALHOST}/api-sileo/v1/ai/langgraph-task-duration-estimation/filter/"

    params = {
        "search_key": "",
        "top": 0,
        "due_date_flag": "Week",
        "sort_field": "-task__date_created",
        "size_per_request": 10,
        "assignee_id": user_profile_pk,
        "workforce_id": 49
    }

    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"
    response = requests.get(full_url, headers=headers)
    response_json = response.json()

    task_names = [
        item["task"]["title"] for item in response_json["data"]["data"]]

    estimates = None

    if task_names:
        estimate_parameters = {
            "user_profile_pk": user_profile_pk,
            "task_names":  json.dumps(task_names),
            "n_similar_task_count": 10
        }
        params = urllib.parse.urlencode(estimate_parameters)
        fetch_estimates = f"{url2}?{params}"
        estimates = requests.get(fetch_estimates, headers=headers)
        estimates = estimates.json()
    # Output response
    return estimates
