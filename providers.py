"""Fabrique de clients : choisit le fournisseur d'API et ses réglages par défaut."""
from __future__ import annotations

import os

from llm import GeminiClient, LLMError, OpenAICompatClient

# max_output_chars : taille maximale d'un résultat d'outil renvoyé au modèle. Les niveaux gratuits
# limitent aussi les tokens par minute : on envoie moins de texte à ces fournisseurs.
PROVIDERS = {
    "gemini": {
        "label": "Google Gemini", "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.8-flash", "max_output_chars": 20_000,
    },
    "groq": {
        "label": "Groq", "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b", "max_output_chars": 6_000,
    },
    "custom": {
        "label": "API compatible OpenAI", "key_env": "LLM_API_KEY", "model_env": "LLM_MODEL",
        "base_env": "LLM_BASE_URL", "default_model": None, "max_output_chars": 6_000,
    },
}


def build_client(provider: str, model: str | None = None, base_url: str | None = None,
                 env=None, http_client=None):
    """Renvoie (client, max_output_chars par défaut). Lève LLMError si la configuration est incomplète."""
    env = os.environ if env is None else env
    if provider not in PROVIDERS:
        raise LLMError(f"Fournisseur inconnu : {provider} (choix : {', '.join(PROVIDERS)}).")
    conf = PROVIDERS[provider]
    key = env.get(conf["key_env"], "").strip()
    model = model or env.get(conf["model_env"], "").strip() or conf["default_model"]
    if not model:
        raise LLMError("Modèle manquant : utilise --model ou renseigne LLM_MODEL dans .env.")

    if provider == "gemini":
        client = GeminiClient(key, model, http_client=http_client)
    else:
        url = base_url or env.get(conf.get("base_env", ""), "").strip() or conf.get("base_url")
        if not url:
            raise LLMError("URL manquante : utilise --base-url ou renseigne LLM_BASE_URL dans .env.")
        client = OpenAICompatClient(key, model, url, label=conf["label"], key_env=conf["key_env"],
                                    model_env=conf["model_env"], http_client=http_client)
    return client, conf["max_output_chars"]
