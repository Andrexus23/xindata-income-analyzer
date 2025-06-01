from pydantic_settings import BaseSettings, JsonConfigSettingsSource
from pydantic import BaseModel, PostgresDsn
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    JsonConfigSettingsSource,
)

class ModelSettings(BaseModel):
    """Набор настроек для взаимодействия с моделью."""

    path: str



class Settings(BaseSettings):

    model_settings: ModelSettings
    model_config = SettingsConfigDict(json_file='config.json')
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """"""
        return (JsonConfigSettingsSource(settings_cls),)


def get_config() -> Settings:
    """Выгрузка конфиг-файла."""
    return Settings()