from lib.sileo.restmodel import Model
import json

TaskCount = Model(namespace="hqzen", resource="task-count", version="v4")
TimeLog = Model(namespace="timelogging", resource="time-log", version="v4")
Card = Model(namespace="board", resource="card-panel", version="v1")
TaskAssignments = Model(namespace="hqzen", resource="task-assignments", version="v4")
LangGraphAITaskEstimation = Model(namespace="ai", resource="langgraph-task-duration-estimation", version="v1")


def fetch_task_counts(args: dict):
    payload = {
        "workforce_id": args.get("workforce_id")
    }
    response = TaskCount.objects.filter(payload)
    return response


def fetch_shift_logs(args: dict):
    payload = {
        "employment_id": args.get("employment_id"),
        "shift_start": args.get("shift_start")
    }
    response = TimeLog.objects.filter(payload)
    return response


def create_new_card(args: dict):
    # TODO: Try to find a way to determine the user's board/column

    form_data = {
        "creator": args.get("creator"),
        "assignees": args.get("assignees"),
        "title": args.get("title"),
        "column": args.get("column", "213"),  # To Do column inside Development Board inside BPOSeats workforce
        "is_public": args.get("is_public", "True"),
    }

    response = Card.objects.create(form_data)
    return response


def fetch_weekly_task_estimates(args: dict):
    user_profile_pk = args.get("user_profile_pk")
    workforce_id = args.get("workforce_id")

    payload = {
        "search_key": "",
        "due_date_flag": "Week",
        "sort_field": "-task__date_created",
        "size_per_request": "10",
        "assignee_id": user_profile_pk,
        "workforce_id": workforce_id
    }

    response = TaskAssignments.objects.filter(payload)
    estimates = None

    try:
        task_names = [
            item["task"]["title"] for item in response["data"]]

        if task_names:
            estimate_parameters = {
                "user_profile_pk": user_profile_pk,
                "task_names":  json.dumps(task_names),
                "n_similar_task_count": 10
            }
            estimates = LangGraphAITaskEstimation.objects.filter(estimate_parameters)

    except Exception as e:
        print(f"Something went wrong! {e}")

    # Output response
    return estimates


def fetch_tasks_due(args: dict):
    user_profile_pk = args.get("user_profile_pk")
    workforce_id = args.get("workforce_id")
    due_date_flag = args.get("due_date_flag")

    payload = {
        "search_key": "",
        "due_date_flag": due_date_flag,
        "sort_field": "-task__date_created",
        "size_per_request": "100",
        "assignee_id": user_profile_pk,
        "workforce_id": workforce_id
    }

    response = TaskAssignments.objects.filter(payload)
    return response
