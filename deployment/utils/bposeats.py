from lib.sileo.restmodel import Model


TaskCount = Model(namespace="hqzen", resource="task-count", version="v4")
TimeLog = Model(namespace="timelogging", resource="time-log", version="v4")
Card = Model(namespace="board", resource="card-panel", version="v1")


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
    form_data = {
        "creator": args.get("user_id"),
        "assignees": args.get("user_id"),
        "title": args.get("title"),
        "column": args.get("column", "213"),
        "is_public": args.get("is_public", True),
    }

    response = Card.objects.create(form_data)
    return response
