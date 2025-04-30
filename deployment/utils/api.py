from lib.sileo.restmodel import Model
from typing import Dict


TaskCounts = Model(namespace="hqzen", resource="task-count", version="v4")


def fetch_task_counts(args: Dict):

    payload = {
        "workforce_id": args["workforce_id"]
    }
    response = TaskCounts.objects.filter(payload)
    return response
