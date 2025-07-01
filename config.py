"""Configuration settings for YouTube Transcription Scraper"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class APIConfig:
    """API configuration settings"""
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: Optional[int] = None
    youtube_api_max_results: int = 50
    
    # Rate limiting settings
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests_per_minute: int = 60


@dataclass
class PlaylistConfig:
    """Playlist configuration settings"""
    default_source_playlist: str = "AI_TO_TRANSCRIBE"
    default_dest_playlist: str = "AI_TRANSCRIBED"


@dataclass
class TranscriptConfig:
    """Transcript processing configuration"""
    preferred_languages: List[str] = None
    max_transcript_length: int = 50000  # characters
    chunk_size: int = 10000  # for processing large transcripts
    
    def __post_init__(self):
        if self.preferred_languages is None:
            self.preferred_languages = ['en', 'es']


@dataclass
class FileConfig:
    """File handling configuration"""
    output_directory: str = "."
    max_filename_length: int = 100
    file_encoding: str = "utf-8"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_output: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig = None
    playlist: PlaylistConfig = None
    transcript: TranscriptConfig = None
    file: FileConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.api is None:
            self.api = APIConfig()
        if self.playlist is None:
            self.playlist = PlaylistConfig()
        if self.transcript is None:
            self.transcript = TranscriptConfig()
        if self.file is None:
            self.file = FileConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


def load_config() -> AppConfig:
    """Load configuration with environment variable overrides"""
    config = AppConfig()
    
    # Override with environment variables if present
    if os.getenv('OPENAI_MODEL'):
        config.api.openai_model = os.getenv('OPENAI_MODEL')
    
    if os.getenv('OPENAI_TEMPERATURE'):
        config.api.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE'))
    
    if os.getenv('MAX_RETRIES'):
        config.api.max_retries = int(os.getenv('MAX_RETRIES'))
    
    if os.getenv('OUTPUT_DIRECTORY'):
        config.file.output_directory = os.getenv('OUTPUT_DIRECTORY')
    
    if os.getenv('LOG_LEVEL'):
        config.logging.level = os.getenv('LOG_LEVEL')
    
    return config


# Global configuration instance
CONFIG = load_config()