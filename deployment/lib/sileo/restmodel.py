import requests
from urllib.parse import urlencode
from functools import wraps


class Defaults:
    headers = {}
    base_url = None
    interceptors = []


class ModelManager:
    def __init__(self, model):
        self.model = model

    def get(self, pk, extras=None):
        url = f"{self.model.base_url}get/{pk}"
        if extras:
            url += f"?{self._parse_get_params(extras)}"

    def filter(self, filters=None, excludes=None):
        url = f"{self.model.base_url}filter/"
        query = self._parse_get_params(filters or {})
        if query:
            url += f"?{query}"
        if excludes:
            url += '&' if query else '?'
            url += self._parse_get_params(excludes)
        return self._fetch("GET", url)

    def _fetch(self, method, url, data=None):
        full_url = f"{Defaults.base_url}{url}" if Defaults.base_url else url
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            **{k: v() if callable(v) else v for k, v in Defaults.headers.items()}
        }
        if method == "POST":
            headers["X-CSRFToken"] = self._get_csrf_token()

        try:
            if method == "POST":
                response = requests.post(full_url, data=data, headers=headers)
            else:
                response = requests.get(full_url, headers=headers)

            if response.status_code in [200, 201]:
                for interceptor in Defaults.interceptors:
                    interceptor(response.json())
                return self.model.callback()(response.json().get("data"))
            elif response.status_code == 403:
                for interceptor in Defaults.interceptors:
                    try:
                        interceptor(response.json())
                    except Exception:
                        interceptor(response.text)
                raise PermissionError("403 Forbidden")
            else:
                response.raise_for_status()

        except Exception as e:
            print("Request error: ", e)
            raise

    # Utility functions
    def _parse_get_params(self, params):
        return urlencode(params)

    def _get_csrf_token(self):
        # For demonstration, CSRF is assumed stored or passed elsewhere
        return "dummy-csrf-token"


class Model:
    def __init__(self, namespace, resource, version="v3", middlewares=None, options=None):
        self.base_url = f"/api-sileo/{version}/{namespace}/{resource}"
        self.middlewares = middlewares or []
        self.options = {"apply_global_middlewares": True}
        if options:
            self.options.update(options)
        self.objects = ModelManager(self)

    def callback(self):
        def wrapper(response):
            if self.options.get("apply_global_middlewares", True):
                response = flow(response, global_middlewares)
            response = flow(response, self.middlewares)
            return response
        return wrapper


global_middlewares = []


def add_global_middleware(*middlewares):
    for middleware in middlewares:
        if callable(middleware):
            global_middlewares.append(middleware)


def flow(data, callbacks):
    for cb in callbacks:
        if isinstance(data, list):
            data = [cb(d) for d in data]
        else:
            data = cb(data)
    return data


def wrap_errors(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print("Wrapped error:", e)
            raise
    return wrapper
