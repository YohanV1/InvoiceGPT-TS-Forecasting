"""LLM client for synthetic data generation"""

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMManager:
    """Manager for LLM instances with rate limiting"""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the LLM manager"""
        self._gemini_flash_client = None
        self._setup_clients()

    def _setup_clients(self):
        """Set up LLM clients with appropriate rate limits"""
        self._gemini_flash_client = None  # Reset to force recreation

    def _create_gemini_client(self) -> ChatGoogleGenerativeAI:
        """Create a Gemini client with appropriate settings"""
        rpm = config.gemini_flash_rate_limit
        
        # Create rate limiter
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=rpm / 60,
            check_every_n_seconds=0.05,
            max_bucket_size=rpm // 2,
        )

        logger.info(f"Using rate limit: {rpm} requests per minute with model {config.gemini_flash_model_name}")
        
        return ChatGoogleGenerativeAI(
            api_key=config.gemini_api_key,
            model=config.gemini_flash_model_name,
            rate_limiter=rate_limiter,
            temperature=config.gemini_temperature,
            top_p=0.95,
            top_k=40,
        )

    @property
    def gemini_flash_client(self) -> ChatGoogleGenerativeAI:
        """Get Gemini flash client with rate limiting"""
        if not self._gemini_flash_client:
            self._gemini_flash_client = self._create_gemini_client()
        return self._gemini_flash_client

    def get_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """Get the LLM client"""
        if not config.gemini_api_key:
            raise ValueError("Google API key not configured")

        try:
            return self.gemini_flash_client
        except Exception as e:
            logger.error(f"Error getting LLM client: {str(e)}")
            return None