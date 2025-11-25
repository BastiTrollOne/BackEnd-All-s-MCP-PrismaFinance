from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LM_STUDIO_URL: str = "http://localhost:1234/v1"
    LLM_MODEL_NAME: str = "qwen/qwen3-4b-2507"

    # Use host.docker.internal, which is the special DNS for containers to reach the host on Windows/Mac
    METABASE_URL_FOR_DOCKER: str = "http://host.docker.internal:32768"
    METABASE_USERNAME: str = "benjamintoledo421@gmail.com"
    METABASE_PASSWORD: str = "asdf123+"

    class Config:
        env_file = ".env"

settings = Settings()
