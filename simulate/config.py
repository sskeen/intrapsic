# ------------------------------------------------------ #
#
#   config.py
#
#   Pipeline configuration for crisis query simulation.
#   Configures LLM parameters, generation settings,
#   and file paths.
#
#   Simone J. Skeen x Claude Code (10-15-2026)
#
# ------------------------------------------------------ #

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Pipeline settings
PIPELINE_CONFIG = {
    'num_queries': 10,              # Queries to generate per persona/seed/model combo
    'random_seed': 56,              # Random seed for reproducibility
}

# Rate limiting
RATE_LIMIT_CONFIG = {
    'delay_seconds': 1.0,           # Delay between API calls
}

# # OpenAI configuration (commented out - using Ollama models only)
# OPENAI_CONFIG = {
#     'model': 'gpt-4o-mini',
#     'temperatures': [0.4, 0.7, 1.0],  # Vary temperature for semantic diversity
#     'max_tokens': 4000,
#     'seed': PIPELINE_CONFIG['random_seed'],
# }

# Temperature values for semantic diversity (shared across all models)
TEMPERATURES = [0.4, 0.7, 1.0]

# Ollama model configurations
OLLAMA_MODELS = [
    {
        'name': 'qwen3:30b',
        'model': 'qwen3:30b',
        'think': True,
    },
    {
        'name': 'gemma4:31b',
        'model': 'gemma4:31b',
        'think': True,
    },
]

# Shared Ollama settings
OLLAMA_BASE_CONFIG = {
    'base_url': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    'num_predict': 4000,
    'seed': PIPELINE_CONFIG['random_seed'],
}

# File paths
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

INPUT_FILES = {
    'personas': DATA_DIR / 'personas.tsv',
    'seed_phrases': DATA_DIR / 'seed_phrases.tsv',
    'persona_context': DATA_DIR / 'persona_context.txt',
    'suicide_risk_levels': DATA_DIR / 'suicide_risk_level.tsv',
}

OUTPUT_FILES = {
    'generated_queries': OUTPUT_DIR / 'generated_queries.tsv',
    'checkpoint': OUTPUT_DIR / '.checkpoint.json',
}

# # Cost estimation (OpenAI pricing per 1M tokens, approximate)
# OPENAI_PRICING = {
#     'input_per_million': 0.15,      # $ per 1M input tokens
#     'output_per_million': 0.60,     # $ per 1M output tokens
# }
