# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    GEMINI_API_KEY: str = Field(...)
    GEMINI_MODEL: str = Field("gemini-2.0-flash")

    GOOGLE_MAPS_API_KEY: str = Field(...)
    PLACES_NEARBY_URL: str = Field("https://places.googleapis.com/v1/places:searchNearby")
    PLACES_FIELD_MASK: str = Field(
        "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount"
    )

    DEFAULT_RADIUS_M: int = Field(2000, ge=200, le=50000)
    DEFAULT_K: int = Field(5, ge=1, le=20)
    DEFAULT_MAX_CANDIDATES: int = Field(25, ge=5, le=60)

settings = Settings()