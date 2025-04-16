from utils.environ import get_settings_variable


POSTGRES_URI = get_settings_variable(
    "POSTGRES_URI",
    default="postgres://bposeatsuser:bposeatspassword@host.docker.internal:5432/bposeats?sslmode=disable",
    required=True
)
API_URL = get_settings_variable(
    "API_URL",
    default="http://host.docker.internal:8000",
    required=True
)
