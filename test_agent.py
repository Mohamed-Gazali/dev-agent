"""Tests de DevAgent (sans réseau). Lancer : python -m unittest -v"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

import agent
from llm import GeminiClient, LLMError, OpenAICompatClient
from providers import build_client
from tools import Workspace


# ---------- helpers ----------
def make_client(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiClient("cle-de-test", "modele-test", http_client=http, **kwargs)


def make_oai(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatClient("cle-de-test", "modele-test", "https://api.exemple.test/v1",
                              http_client=http, **kwargs)


def reply(parts, role="model"):  # réponse au format Gemini
    return httpx.Response(200, json={"candidates": [{"content": {"role": role, "parts": parts}}]})


def oai_reply(content=None, tool_calls=None):  # réponse au format OpenAI
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": {"total_tokens": 42}})


def oai_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class TempProject(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "main.py").write_text("print('bonjour')\nx = 1\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()


# ---------- outils ----------
class WorkspaceTests(TempProject):
    def setUp(self):
        super().setUp()
        (self.root / ".env").write_text("SECRET=abc", encoding="utf-8")
        (self.root / ".env.example").write_text("SECRET=", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "lib.js").write_text("bonjour", encoding="utf-8")
        self.answers = []
        self.previews = []

        def confirm(rel, preview):
            self.previews.append((rel, preview))
            return self.answers.pop(0) if self.answers else False

        self.ws = Workspace(self.root, confirm=confirm)

    def test_path_traversal_is_blocked(self):
        for bad in ["../secret.txt", "..", "/etc/passwd", "C:\\Windows\\win.ini"]:
            res = self.ws.run("read_file", {"path": bad})
            self.assertIn("error", res, bad)

    def test_colon_and_null_in_paths_are_rejected(self):
        for bad in [".env::$DATA", "main.py:flux", "C:\\x.txt", "a\x00b"]:
            res = self.ws.run("read_file", {"path": bad})
            self.assertIn("interdits", res.get("error", ""), bad)

    def test_secret_files_are_hidden_and_unreadable(self):
        listing = self.ws.run("list_files", {})["result"]
        self.assertNotIn(".env\n", listing + "\n")
        self.assertIn(".env.example", listing)
        self.assertNotIn("node_modules", listing)
        self.assertIn("error", self.ws.run("read_file", {"path": ".env"}))
        self.assertIn("error", self.ws.run("write_file", {"path": ".env", "content": "X=1"}))
        found = self.ws.run("search_text", {"query": "SECRET"})["result"]
        self.assertNotIn("abc", found)

    def test_read_and_search(self):
        text = self.ws.run("read_file", {"path": "main.py"})["result"]
        self.assertIn("    1| print('bonjour')", text)
        found = self.ws.run("search_text", {"query": "BONJOUR"})["result"]
        self.assertIn("main.py:1:", found)
        self.assertNotIn("node_modules", found)

    def test_write_needs_confirmation(self):
        self.answers = [False]
        res = self.ws.run("write_file", {"path": "main.py", "content": "print('salut')\n"})
        self.assertIn("error", res)
        self.assertEqual((self.root / "main.py").read_text(encoding="utf-8"), "print('bonjour')\nx = 1\n")
        self.assertIn("-print('bonjour')", self.previews[-1][1])
        self.assertIn("+print('salut')", self.previews[-1][1])

        self.answers = [True]
        res = self.ws.run("write_file", {"path": "sous/dossier/new.py", "content": "a = 1\n"})
        self.assertIn("result", res)
        self.assertEqual((self.root / "sous" / "dossier" / "new.py").read_text(encoding="utf-8"), "a = 1\n")

    def test_write_cannot_touch_git_or_outside(self):
        self.answers = [True, True]
        self.assertIn("error", self.ws.run("write_file", {"path": ".git/config", "content": "x"}))
        self.assertIn("error", self.ws.run("write_file", {"path": "../evil.py", "content": "x"}))

    def test_crlf_is_preserved(self):
        (self.root / "win.txt").write_bytes(b"a\r\nb\r\n")
        self.answers = [True]
        self.ws.run("write_file", {"path": "win.txt", "content": "a\nc\n"})
        self.assertEqual((self.root / "win.txt").read_bytes(), b"a\r\nc\r\n")

    def test_unknown_tool_and_bad_args(self):
        self.assertIn("error", self.ws.run("rm_rf", {}))
        self.assertIn("error", self.ws.run("read_file", {"chemin": "main.py"}))

    def test_output_limit_is_configurable(self):
        (self.root / "gros.txt").write_text("x" * 5000, encoding="utf-8")
        small = Workspace(self.root, max_output_chars=100)
        text = small.run("read_file", {"path": "gros.txt"})["result"]
        self.assertLess(len(text), 200)
        self.assertIn("tronquée", text)


# ---------- boucle d'agent (Gemini) ----------
class AgentLoopTests(TempProject):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.root, confirm=lambda rel, preview: True)

    def test_tool_call_then_final_answer(self):
        seen = []
        call_part = {"functionCall": {"name": "list_files", "args": {}, "id": "call-1"},
                     "thoughtSignature": "SIG-OPAQUE"}

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            self.assertEqual(request.headers["x-goog-api-key"], "cle-de-test")
            return reply([call_part]) if len(seen) == 1 else reply([{"text": "Il y a main.py."}])

        history = []
        answer = agent.run_turn(make_client(handler), self.ws, history, "Que contient le projet ?",
                                out=lambda *_: None)
        self.assertEqual(answer, "Il y a main.py.")
        self.assertEqual(history[1]["parts"][0]["thoughtSignature"], "SIG-OPAQUE")
        fr = history[2]["parts"][0]["functionResponse"]
        self.assertEqual((fr["name"], fr["id"]), ("list_files", "call-1"))
        self.assertIn("main.py", fr["response"]["result"])
        self.assertEqual(len(seen[1]["contents"]), 3)

    def test_step_limit_keeps_history_valid(self):
        def handler(request):
            return reply([{"functionCall": {"name": "list_files", "args": {}}}])

        history = []
        answer = agent.run_turn(make_client(handler), self.ws, history, "boucle", max_steps=3,
                                out=lambda *_: None)
        self.assertIn("Limite de 3 étapes", answer)
        self.assertEqual(history[-1]["role"], "model")

    def test_error_rolls_back_history(self):
        def handler(request):
            return httpx.Response(404, json={"error": {"message": "model not found"}})

        history = [{"role": "user", "parts": [{"text": "avant"}]}]
        with self.assertRaises(LLMError) as ctx:
            agent.run_turn(make_client(handler), self.ws, history, "test", out=lambda *_: None)
        self.assertIn("GEMINI_MODEL", str(ctx.exception))
        self.assertEqual(len(history), 1)

    def test_empty_candidates_raise_error_and_roll_back(self):
        def handler(request):
            return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

        history = []
        with self.assertRaises(LLMError):
            agent.run_turn(make_client(handler), self.ws, history, "x", out=lambda *_: None)
        self.assertEqual(history, [])

    def test_retry_on_429_then_success(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "quota"}})
            return reply([{"text": "ok"}])

        with mock.patch("llm.time.sleep") as sleep:
            answer = agent.run_turn(make_client(handler), self.ws, [], "x", out=lambda *_: None)
        self.assertEqual(answer, "ok")
        sleep.assert_called_once_with(1.0)

    def test_missing_key_is_explained(self):
        with self.assertRaises(LLMError):
            GeminiClient("colle_ta_cle_ici", "m")

    def test_first_message_contains_file_listing(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return reply([{"text": "ok"}])

        history = []
        client = make_client(handler)
        agent.run_turn(client, self.ws, history, "Que fait main.py ?", out=lambda *_: None)
        first = seen[0]["contents"][0]["parts"][0]["text"]
        self.assertIn("main.py", first)
        self.assertIn("Que fait main.py ?", first)
        agent.run_turn(client, self.ws, history, "Merci", out=lambda *_: None)
        self.assertEqual(seen[1]["contents"][-1]["parts"][0]["text"], "Merci")


# ---------- quotas et réseau (logique commune) ----------
class QuotaTests(unittest.TestCase):
    def test_daily_quota_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            body = {"error": {"message": "quota", "details": [
                {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
            return httpx.Response(429, json=body)

        client = make_client(handler)
        with mock.patch("llm.time.sleep") as sleep:
            with self.assertRaises(LLMError) as ctx:
                client.generate_raw([], "s", [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.request_count, 1)
        sleep.assert_not_called()
        self.assertIn("JOURNALIER", str(ctx.exception))
        self.assertIn("--provider", str(ctx.exception))

    def test_retry_delay_from_body_is_used(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, json={"error": {"message": "rpm", "details": [{"retryDelay": "7s"}]}})
            return reply([{"text": "ok"}])

        client = make_client(handler)
        with mock.patch("llm.time.sleep") as sleep:
            client.generate_raw([], "s", [])
        sleep.assert_called_once_with(7.0)
        self.assertEqual(client.request_count, 2)

    def test_timeout_is_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ReadTimeout("trop lent", request=request)
            return reply([{"text": "ok"}])

        client = make_client(handler)
        with mock.patch("llm.time.sleep"):
            data = client.generate_raw([], "s", [])
        self.assertEqual(data["candidates"][0]["content"]["parts"][0]["text"], "ok")
        self.assertEqual(len(calls), 2)

    def test_too_large_request_has_a_clear_hint(self):
        def handler(request):
            return httpx.Response(413, json={"error": {"message": "Request too large for model"}})

        with self.assertRaises(LLMError) as ctx:
            make_oai(handler).generate_raw([], "s", [])
        self.assertIn("--max-output-chars", str(ctx.exception))


# ---------- API compatibles OpenAI (Groq, ...) ----------
class OpenAICompatTests(TempProject):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.root, confirm=lambda rel, preview: True)

    def test_tool_roundtrip_uses_openai_format(self):
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return oai_reply(None, [oai_call("list_files", "{}")])
            return oai_reply("Il y a main.py.")

        history = []
        answer = agent.run_turn(make_oai(handler), self.ws, history, "Que contient le projet ?",
                                out=lambda *_: None)
        self.assertEqual(answer, "Il y a main.py.")

        first = seen[0]
        self.assertTrue(str(first.url).endswith("/v1/chat/completions"))
        self.assertEqual(first.headers["authorization"], "Bearer cle-de-test")
        body = json.loads(first.content)
        self.assertEqual(body["model"], "modele-test")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertIn("name", body["tools"][0]["function"])

        second = json.loads(seen[1].content)["messages"]
        roles = [m["role"] for m in second]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertEqual(second[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second[3]["tool_call_id"], "call_1")
        self.assertNotIn("name", second[3])
        self.assertIn("main.py", second[3]["content"])

    def test_invalid_json_arguments_become_a_tool_error(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            if len(seen) == 1:
                return oai_reply(None, [oai_call("read_file", "{pas du json")])
            return oai_reply("Désolé.")

        history = []
        agent.run_turn(make_oai(handler), self.ws, history, "lis", out=lambda *_: None)
        tool_message = seen[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn("Arguments invalides", tool_message["content"])

    def test_write_goes_through_confirmation(self):
        decisions = []
        ws = Workspace(self.root, confirm=lambda rel, preview: decisions.append(rel) or False)
        seen = []

        def handler(request):
            seen.append(1)
            if len(seen) == 1:
                args = json.dumps({"path": "main.py", "content": "x = 2\n"})
                return oai_reply(None, [oai_call("write_file", args)])
            return oai_reply("Refusé par l'utilisateur.")

        agent.run_turn(make_oai(handler), ws, [], "modifie", out=lambda *_: None)
        self.assertEqual(decisions, ["main.py"])
        self.assertEqual((self.root / "main.py").read_text(encoding="utf-8"), "print('bonjour')\nx = 1\n")

    def test_daily_quota_message_from_groq_style_error(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(429, json={"error": {"message":
                "Rate limit reached for model on requests per day (RPD): Limit 1000, Used 1000"}})

        client = make_oai(handler)
        with mock.patch("llm.time.sleep") as sleep:
            with self.assertRaises(LLMError) as ctx:
                client.generate([], "s", [])
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()
        self.assertIn("JOURNALIER", str(ctx.exception))

    def test_empty_choices_raise_error(self):
        def handler(request):
            return httpx.Response(200, json={"choices": []})

        with self.assertRaises(LLMError):
            make_oai(handler).generate([], "s", [])


# ---------- choix du fournisseur ----------
class ProviderTests(unittest.TestCase):
    def test_groq_defaults(self):
        client, chars = build_client("groq", env={"GROQ_API_KEY": "gsk_test"})
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertEqual(client.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(client.model, "openai/gpt-oss-120b")
        self.assertEqual(chars, 6000)

    def test_model_from_env_and_cli_priority(self):
        env = {"GROQ_API_KEY": "gsk_test", "GROQ_MODEL": "modele-env"}
        self.assertEqual(build_client("groq", env=env)[0].model, "modele-env")
        self.assertEqual(build_client("groq", model="modele-cli", env=env)[0].model, "modele-cli")

    def test_gemini(self):
        client, chars = build_client("gemini", env={"GEMINI_API_KEY": "AIza-test"})
        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(chars, 20000)

    def test_custom_needs_url_and_model(self):
        with self.assertRaises(LLMError):
            build_client("custom", env={"LLM_API_KEY": "k", "LLM_MODEL": "m"})
        with self.assertRaises(LLMError):
            build_client("custom", base_url="https://x.test/v1", env={"LLM_API_KEY": "k"})
        client, _ = build_client("custom", model="m", base_url="https://x.test/v1/", env={"LLM_API_KEY": "k"})
        self.assertEqual(client.base_url, "https://x.test/v1")

    def test_missing_or_placeholder_key(self):
        with self.assertRaises(LLMError):
            build_client("groq", env={})
        with self.assertRaises(LLMError):
            build_client("groq", env={"GROQ_API_KEY": "colle_ta_cle_ici"})

    def test_unknown_provider(self):
        with self.assertRaises(LLMError):
            build_client("inconnu", env={})


if __name__ == "__main__":
    unittest.main()
