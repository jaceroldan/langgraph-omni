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
