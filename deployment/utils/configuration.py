import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    auth_token: str
    thread_id: str
    model_name: str = "gpt-4o"

    user_profile_pk: str = ""
    employment_id: str = ""
    company_id: str = ""
    payroll_id: str = ""
    source: Optional[str] = None
    workforce_id: Optional[str] = None
    job_position: Optional[str] = None
    x_timezone: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
