"""Clients d'API derrière une interface commune : Gemini et API « compatibles OpenAI » (Groq, etc.).

Chaque client sait :
- envoyer l'historique + la liste des outils et renvoyer une réponse normalisée (`Reply`) ;
- fabriquer les messages dans SON format natif (l'historique reste dans le format du fournisseur).
La logique commune (nouvelles tentatives, quotas, erreurs lisibles) vit dans `HttpLLM`.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(Exception):
    """Erreur côté API (clé, quota, modèle, réseau, réponse inutilisable...)."""


@dataclass
class ToolCall:
    id: str | None
    name: str
    args: dict
    error: str | None = None  # arguments illisibles envoyés par le modèle


@dataclass
class Reply:
    text: str
    calls: list = field(default_factory=list)
    messages: list = field(default_factory=list)  # messages natifs à ajouter à l'historique
    tokens: int | None = None


def check_key(api_key: str, env_name: str) -> None:
    if not api_key or api_key.startswith("colle_"):
        raise LLMError(f"{env_name} manquante : renseigne ta clé dans le fichier .env.")


class HttpLLM:
    label = "API"
    model_env = "--model"

    def __init__(self, model: str, http_client: httpx.Client | None = None, max_retries: int = 2):
        self.model = model
        self.max_retries = max_retries
        self.http = http_client or httpx.Client(timeout=120)
        self.request_count = 0  # requêtes envoyées depuis le lancement (nouvelles tentatives comprises)

    # ---------- HTTP commun ----------
    def _post(self, url: str, headers: dict, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            try:
                resp = self.http.post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise LLMError("L'API ne répond pas (délai dépassé) après plusieurs tentatives.") from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Connexion impossible à l'API : {exc}") from exc

            if resp.status_code == 200:
                return resp.json()

            # Quota JOURNALIER épuisé : réessayer est inutile et consomme encore du quota.
            if resp.status_code == 429 and self._is_daily_quota(resp):
                raise LLMError(self._explain(resp))

            retryable = resp.status_code in (429, 500, 502, 503, 504)
            if retryable and attempt < self.max_retries:
                time.sleep(self._wait_seconds(resp, attempt))
                continue
            raise LLMError(self._explain(resp))
        raise LLMError("Échec après plusieurs tentatives.")  # pragma: no cover

    @staticmethod
    def _is_daily_quota(resp: httpx.Response) -> bool:
        low = resp.text.lower()
        return any(key in low for key in ("perday", "per day", "per-day", "daily", "(rpd)", "(tpd)"))

    @staticmethod
    def _wait_seconds(resp: httpx.Response, attempt: int) -> float:
        try:
            return min(60.0, float(resp.headers.get("retry-after", "")))
        except ValueError:
            pass
        match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', resp.text)
        if match:
            return min(60.0, float(match.group(1)))
        return 10.0 * (attempt + 1)

    def _explain(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            err = data.get("error", data) if isinstance(data, dict) else data
            detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except Exception:
            detail = resp.text
        detail = (detail or "").strip().replace("\n", " ")[:300]
        code = resp.status_code
        low = resp.text.lower()
        if code in (401, 403):
            hint = "Clé invalide, expirée, ou accès refusé (vérifie la clé et le projet)."
        elif code == 404:
            hint = f"Modèle '{self.model}' introuvable : change le modèle ({self.model_env})."
        elif code == 413 or (code == 429 and "tokens per minute" in low):
            hint = ("Requête trop volumineuse pour la limite de tokens par minute : tape /reset "
                    "ou baisse --max-output-chars.")
        elif code == 429 and self._is_daily_quota(resp):
            hint = (f"Quota JOURNALIER atteint pour le modèle '{self.model}' (chez {self.label}). "
                    "Relance avec --model <autre modèle> ou --provider <autre fournisseur>, "
                    "ou attends la réinitialisation.")
        elif code == 429:
            hint = "Limite par minute atteinte : attends une minute puis réessaie."
        elif code == 400 and "tool_use_failed" in low:
            hint = "Le modèle a mal formé un appel d'outil : réessaie, ou change de modèle."
        elif code == 400:
            hint = "Requête refusée par l'API."
        else:
            hint = "Erreur du service."
        return f"[HTTP {code}] {hint} Détail : {detail}"


# ======================================================================
# Gemini (Google AI Studio) : méthode generateContent
# ======================================================================
class GeminiClient(HttpLLM):
    label = "Google Gemini"
    model_env = "--model, ou GEMINI_MODEL dans .env"

    def __init__(self, api_key: str, model: str, http_client: httpx.Client | None = None,
                 max_retries: int = 2):
        check_key(api_key, "GEMINI_API_KEY")
        super().__init__(model, http_client, max_retries)
        self.api_key = api_key

    # messages natifs
    def user_message(self, text: str) -> list:
        return [{"role": "user", "parts": [{"text": text}]}]

    def assistant_text(self, text: str) -> list:
        return [{"role": "model", "parts": [{"text": text}]}]

    def tool_result_messages(self, results: list) -> list:
        parts = []
        for call, result in results:
            response = {"name": call.name, "response": result}
            if call.id:
                response["id"] = call.id
            parts.append({"functionResponse": response})
        return [{"role": "user", "parts": parts}]

    # appel
    def generate_raw(self, contents: list, system_instruction: str, tool_declarations: list) -> dict:
        url = f"{GEMINI_ROOT}/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "tools": [{"functionDeclarations": tool_declarations}],
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        return self._post(url, headers, body)

    def generate(self, history: list, system_instruction: str, tool_declarations: list) -> Reply:
        data = self.generate_raw(history, system_instruction, tool_declarations)
        candidates = data.get("candidates") or []
        if not candidates or "content" not in candidates[0]:
            reason = (data.get("promptFeedback", {}).get("blockReason")
                      or (candidates[0].get("finishReason") if candidates else None)
                      or "réponse vide")
            raise LLMError(f"Réponse inutilisable de l'API ({reason}).")
        content = candidates[0]["content"]  # conservé tel quel (signatures de réflexion incluses)
        parts = content.get("parts", [])
        calls = [ToolCall(p["functionCall"].get("id"), p["functionCall"].get("name", ""),
                          p["functionCall"].get("args") or {})
                 for p in parts if "functionCall" in p]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        tokens = data.get("usageMetadata", {}).get("totalTokenCount")
        return Reply(text, calls, [content], tokens)


# ======================================================================
# API « compatibles OpenAI » : /chat/completions (Groq, OpenRouter, Mistral, ...)
# ======================================================================
class OpenAICompatClient(HttpLLM):
    def __init__(self, api_key: str, model: str, base_url: str, label: str = "API compatible OpenAI",
                 key_env: str = "LLM_API_KEY", model_env: str = "--model",
                 http_client: httpx.Client | None = None, max_retries: int = 2):
        check_key(api_key, key_env)
        super().__init__(model, http_client, max_retries)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.label = label
        self.model_env = f"--model, ou {model_env} dans .env"

    # messages natifs
    def user_message(self, text: str) -> list:
        return [{"role": "user", "content": text}]

    def assistant_text(self, text: str) -> list:
        return [{"role": "assistant", "content": text}]

    def tool_result_messages(self, results: list) -> list:
        # Pas de champ "name" : certains fournisseurs (Groq) le refusent.
        return [{"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(result, ensure_ascii=False)} for call, result in results]

    @staticmethod
    def _tools(declarations: list) -> list:
        return [{"type": "function",
                 "function": {"name": d["name"], "description": d["description"],
                              "parameters": d["parameters"]}} for d in declarations]

    # appel
    def generate_raw(self, messages: list, system_instruction: str, tool_declarations: list) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_instruction}] + messages,
            "tools": self._tools(tool_declarations),
            "tool_choice": "auto",
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return self._post(f"{self.base_url}/chat/completions", headers, body)

    def generate(self, history: list, system_instruction: str, tool_declarations: list) -> Reply:
        data = self.generate_raw(history, system_instruction, tool_declarations)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise LLMError("Réponse inutilisable de l'API (aucun message).")
        message = choices[0]["message"]
        text = message.get("content") or ""

        calls, native_calls = [], []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            name = function.get("name", "")
            raw_args = function.get("arguments") or "{}"
            error, args = None, {}
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    error = "les arguments doivent être un objet JSON"
            except ValueError:
                error = "arguments JSON illisibles"
            call_id = raw_call.get("id") or f"call_{index}"
            calls.append(ToolCall(call_id, name, args, error))
            native_calls.append({
                "id": call_id, "type": "function",
                "function": {"name": name,
                             "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args)},
            })

        # Message reconstruit « proprement » : on ne renvoie pas les champs propres au fournisseur.
        native = {"role": "assistant", "content": text}
        if native_calls:
            native["tool_calls"] = native_calls
        tokens = (data.get("usage") or {}).get("total_tokens")
        return Reply(text, calls, [native], tokens)
