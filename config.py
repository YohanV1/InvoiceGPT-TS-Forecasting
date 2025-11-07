"""Configuration for synthetic data generation"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Model Names
    gemini_flash_model_name = os.getenv("GEMINI_FLASH_MODEL_NAME", "gemini-2.0-flash")
    
    # Model Configuration
    gemini_temperature = float(os.getenv("GEMINI_TEMPERATURE", 1.0))
    
    # Output Configuration
    output_dir = os.getenv("OUTPUT_DIR", "generated_data")
    num_samples_per_category = int(os.getenv("SAMPLES_PER_CATEGORY", 10))  # Change to 120 for production
    
    # Rate limits (requests per minute)
    gemini_flash_rate_limit = int(os.getenv("GEMINI_FLASH_RATE_LIMIT", 1000))

config = Config()

