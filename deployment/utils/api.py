from lib.sileo.restmodel import Model


TaskCount = Model(namespace="hqzen", resource="task-count", version="v4")
TimeLog = Model(namespace="timelogging", resource="time-log", version="v4")


def fetch_task_counts(args: dict):
    payload = {
        "workforce_id": args["workforce_id"]
    }
    response = TaskCount.objects.filter(payload)
    return response


def fetch_shift_logs(args: dict):
    payload = {
        "employment_id": args["employment_id"],
        "shift_start": args["shift_start"]
    }
    response = TimeLog.objects.filter(payload)
    return response
