# ------------------------------------------------------ #
#
#   llm_clients.py
#
#   LLM client wrappers for OpenAI and Ollama.
#   Creates fresh client instances per generation call.
#   Includes rate limiting and random seed support.
#
#   Simone J. Skeen x Claude Code (09-30-2026)
#
# ------------------------------------------------------ #

import time


def normalize_text(text: str) -> str:
    """
    Normalize Unicode characters to ASCII-friendly equivalents.
    Replaces curly quotes, em-dashes, mojibake artifacts, and other special characters.
    """
    # Fix common mojibake (double-encoded UTF-8) patterns first
    mojibake_fixes = {
        '‚Äô': "'",      # right single quote
        '‚Äò': "'",      # left single quote
        '‚Äù': '"',      # right double quote
        '‚Äú': '"',      # left double quote
        '‚Äî': '-',      # em-dash
        '‚Äì': '-',      # en-dash
        '‚Ä¶': '...',    # ellipsis
        '‚Ä†': ' ',      # non-breaking space
        'Äî': '-',       # em-dash variant
        'Äô': "'",       # quote variant
        'Äù': '"',       # quote variant
        'Äú': '"',       # quote variant
    }
    for pattern, replacement in mojibake_fixes.items():
        text = text.replace(pattern, replacement)

    # Then fix proper Unicode characters
    unicode_replacements = {
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2013': '-',   # en-dash
        '\u2014': '-',   # em-dash
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',   # non-breaking space
        '\u2022': '-',   # bullet
        '\u2032': "'",   # prime
        '\u2033': '"',   # double prime
        '\u00b4': "'",   # acute accent
        '\u0060': "'",   # grave accent
        '\u00ab': '"',   # left guillemet
        '\u00bb': '"',   # right guillemet
        '\u201a': "'",   # single low quote
        '\u201e': '"',   # double low quote
    }
    for char, replacement in unicode_replacements.items():
        text = text.replace(char, replacement)

    # Strip any remaining non-ASCII characters (silently drop them)
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text

# from openai import OpenAI  # Commented out - using Ollama models only
from ollama import Client as OllamaClient

from config import OLLAMA_BASE_CONFIG, RATE_LIMIT_CONFIG
# from config import OPENAI_CONFIG  # Commented out - using Ollama models only


def get_openai_response(prompt: str, config: dict = None) -> dict:
    """
    Generate a response using OpenAI API with a fresh client instance.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    config : dict, optional
        Override default OPENAI_CONFIG settings. Use 'temperature' key
        to specify temperature for this call.

    Returns
    -------
    dict
        Response containing 'text', 'token_count', 'model', and 'temperature' keys
    """
    cfg = {**OPENAI_CONFIG, **(config or {})}

    # Fresh client instance per call
    client = OpenAI()

    # Use temperature from config override, or default to first in list
    temperature = cfg.get('temperature', cfg.get('temperatures', [0.7])[0])

    response = client.chat.completions.create(
        model=cfg['model'],
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=cfg['max_tokens'],
        temperature=temperature,
        seed=cfg.get('seed'),
    )

    # Rate limiting: delay after each call to avoid throttling
    delay = RATE_LIMIT_CONFIG.get('delay_seconds', 1.0)
    time.sleep(delay)

    message = response.choices[0].message
    text = message.content or ''

    return {
        'text': normalize_text(text),
        'token_count': response.usage.completion_tokens,
        'model': cfg['model'],
        'temperature': temperature,
    }


def get_ollama_response(prompt: str, config: dict = None) -> dict:
    """
    Generate a response using Ollama with a fresh client instance.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    config : dict, optional
        Override default OLLAMA_BASE_CONFIG settings. Must include 'model'
        and 'temperature' keys for the specific model to use.

    Returns
    -------
    dict
        Response containing 'text', 'token_count', 'model', and 'temperature' keys
    """
    cfg = {**OLLAMA_BASE_CONFIG, **(config or {})}
    temperature = cfg['temperature']

    # Fresh client instance per call
    client = OllamaClient(host=cfg['base_url'])

    response = client.generate(
        model=cfg['model'],
        prompt=prompt,
        options={
            'temperature': temperature,
            'num_predict': cfg['num_predict'],
            'seed': cfg.get('seed', 56),
        },
        think=cfg.get('think', False),
    )

    # Extract text, excluding thinking tags if present
    raw_text = response.get('response', '')
    text = raw_text

    if cfg.get('think') and '</think>' in text:
        # Extract content AFTER the closing </think> tag
        parts = text.split('</think>', 1)
        if len(parts) > 1:
            text = parts[1].strip()
        else:
            text = ''

    return {
        'text': normalize_text(text),
        'token_count': response.get('eval_count', 0),
        'model': cfg['model'],
        'temperature': temperature,
    }


def estimate_openai_cost(
    num_generations: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 150,
    pricing: dict = None,
) -> dict:
    """
    Estimate OpenAI API cost for a batch of generations.

    Parameters
    ----------
    num_generations : int
        Number of API calls to estimate
    avg_input_tokens : int
        Average input tokens per call
    avg_output_tokens : int
        Average output tokens per call
    pricing : dict, optional
        Override default pricing with 'input_per_million' and 'output_per_million'

    Returns
    -------
    dict
        Cost breakdown with 'input_cost', 'output_cost', and 'total_cost'
    """
    from config import OPENAI_PRICING
    price = pricing or OPENAI_PRICING

    total_input = num_generations * avg_input_tokens
    total_output = num_generations * avg_output_tokens

    input_cost = (total_input / 1_000_000) * price['input_per_million']
    output_cost = (total_output / 1_000_000) * price['output_per_million']

    return {
        'num_generations': num_generations,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost,
    }
