"""
Prompt templates for zero-shot and domain-specific sentiment classification.

Each builder returns a list of message dicts for the Ollama chat API.
Valid output labels: positive / neutral / negative
"""

_SYSTEM_ZERO_SHOT = (
    "You are a sentiment classifier. "
    "Respond with exactly one word: positive, neutral, or negative."
)

_SYSTEM_NYT = (
    "You are analyzing formal editorial and opinion writing from major newspapers. "
    "These sentences may contain hedging language, nuanced argumentation, or subtle critique. "
    "Classify the sentiment of the sentence as exactly one word: positive, neutral, or negative."
)

_SYSTEM_AMAZON = (
    "You are analyzing informal product reviews written by online shoppers. "
    "These texts often contain slang, sarcasm, mixed opinions, and colloquial expressions. "
    "Classify the overall sentiment of the review as exactly one word: positive, neutral, or negative."
)


def build_prompt(text: str, mode: str, domain: str) -> list[dict]:
    """
    mode:   'zero_shot' | 'domain_specific'
    domain: 'nyt' | 'amazon'
    """
    if mode == "zero_shot":
        system = _SYSTEM_ZERO_SHOT
    elif mode == "domain_specific":
        system = _SYSTEM_NYT if domain == "nyt" else _SYSTEM_AMAZON
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
