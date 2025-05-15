"""
This util file contains utility function for getting data from environment
variables


"""

from urllib.parse import urlparse, urlunparse, quote
import os


USE_STRICT = os.environ.get('strict_config', 'False') == 'True'


# COPIED FROM bposeats/src/plex/plex/core/utils/environ.py
def get_settings_variable(env_name, default, required=False, parser=None):
    """ This function allows getting values of environment variables

    Arguments:
        * env_name - the name of the environment variable
        * default - the default value if the environment variable is not set
        * required - a boolean, set to True if you want to raise an exception
            if the environment variable is not defined
        * parser - an optional function that will process the value of the
            environment variable, this can be used to say convert a json string
            to a dict by passing in `dict`
    """
    env_value = os.environ.get(env_name, None)
    if required is True and USE_STRICT is True and env_value is None:
        raise ValueError('The environment variable {} is not defined'.format(
            env_name))
    env_value = env_value if env_value is not None else default
    if parser is not None:
        env_value = parser(env_value)
    return env_value


def replace_postgres_hostname(uri: str, new_hostname: str) -> str:
    parsed = urlparse(uri)
    user = parsed.username
    password = parsed.password or ''
    port = f":{parsed.port}" if parsed.port else ''
    auth = f"{user}:{quote(password)}" if password else user
    netloc = f"{auth}@{new_hostname}{port}"
    return urlunparse(parsed._replace(netloc=netloc))
