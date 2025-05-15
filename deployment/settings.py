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

# Memory settings
MODEL_HISTORY_LENGTH = 4  # Do not use Odd numbers or 400 error occurs "Invalid parameter"
TOKEN_LIMIT = 2000
TOKEN_LIMIT_SMALL = 500
TOKEN_LIMIT_LARGE = 6000

# PGVECTOR
PGVECTOR_CONNECTION_STRING = get_settings_variable(
    "PGVECTOR_CONNECTION_STRING",
    default="postgresql+psycopg2://bposeatsuser:bposeatspassword@host.docker.internal:5432/bposeats",
    required=True
)
COLLECTION_NAME = "recall_memories"

# HQZEN NAVIGATION LINKS
SITE_DOMAINS = {
    "applybpo.com": get_settings_variable(
        "APPLYBPO_URL", default="https://applybpo.com", required=True
    ),
    "bposeats.com": get_settings_variable(
        "BPOSEATS_URL", default="https://bposeats.com", required=True
    ),
    "bpotube.com": get_settings_variable(
        "BPOTUBE_URL", default="https://bpotube.com", required=True
    ),
    "centralbpo.com": get_settings_variable(
        "CENTRALBPO_URL", default="https://centralbpo.com", required=True
    ),
    "scalema.com": get_settings_variable(
        "SCALEMA_URL", default="https://scalema.com", required=True
    ),
    "hqzen.com": get_settings_variable(
        "HQZEN_URL", default="https://hqzen.com", required=True
    ),
}
