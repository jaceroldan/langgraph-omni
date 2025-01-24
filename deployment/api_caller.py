import requests
import urllib.parse

VITE_LOCALHOST="http://host.docker.internal:8000"
# VITE_LOCALHOST="http://0.0.0.0:8000" # Only used for testing prompts



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

# AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl9wayI6MTM1OTc0NSwiZXhwIjoxNzM3OTQ4NjE4fQ.21b-WmiCUtKaONOto5RxojOqWuZegD8jnu6GRtLrFN0"

# print(fetch_shift_logs(
#     auth_token=AUTH_TOKEN,
#     employment_id="7047",
#     shift_start="2025-01-20T00:00:00+08:00"
# ))
