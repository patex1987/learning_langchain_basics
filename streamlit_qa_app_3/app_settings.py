from pydantic_settings import BaseSettings


class StreamlitAppSettings(BaseSettings):
    google_api_key: str
    faq_file_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
