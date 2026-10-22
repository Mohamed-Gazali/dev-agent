"""Client minimal pour l'API Gemini (REST, méthode generateContent).

Un seul rôle : envoyer l'historique + la liste des outils, renvoyer la réponse JSON,
et transformer les erreurs HTTP en messages compréhensibles.
"""
from __future__ import annotations

import re
import time

import httpx

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(Exception):
    """Erreur côté API (clé, quota, modèle, réseau...)."""


class GeminiClient:
    def __init__(self, api_key: str, model: str, http_client: httpx.Client | None = None,
                 max_retries: int = 2):
        if not api_key or api_key.startswith("colle_"):
            raise LLMError("GEMINI_API_KEY manquante : renseigne ta clé dans le fichier .env.")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.http = http_client or httpx.Client(timeout=120)
        self.request_count = 0  # requêtes envoyées à l'API depuis le lancement (retries compris)

    def generate(self, contents: list, system_instruction: str, tool_declarations: list) -> dict:
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "tools": [{"functionDeclarations": tool_declarations}],
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

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
        text = resp.text
        return "PerDay" in text or "per day" in text.lower() or "daily" in text.lower()

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
            detail = resp.json()["error"]["message"]
        except Exception:
            detail = resp.text
        detail = (detail or "").strip().replace("\n", " ")[:300]
        code = resp.status_code
        if code in (401, 403):
            hint = "Clé invalide, expirée, ou accès refusé (vérifie la clé et le projet)."
        elif code == 404:
            hint = f"Modèle '{self.model}' introuvable : change GEMINI_MODEL dans .env."
        elif code == 429 and self._is_daily_quota(resp):
            hint = (f"Quota JOURNALIER du niveau gratuit atteint pour le modèle '{self.model}' "
                    "(remise à zéro à minuit, heure du Pacifique). Relance avec --model <autre modèle> "
                    "(par exemple gemini-3.6-flash) ou attends la réinitialisation.")
        elif code == 429:
            hint = "Limite par minute atteinte : attends une minute puis réessaie."
        elif code == 400:
            hint = "Requête refusée par l'API."
        else:
            hint = "Erreur du service."
        return f"[HTTP {code}] {hint} Détail : {detail}"
