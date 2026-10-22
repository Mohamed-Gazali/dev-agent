"""Outils de l'agent : lecture, recherche et écriture dans UN dossier de projet.

Les règles de sécurité sont appliquées ICI, dans le code, et pas seulement dans le prompt :
- aucun accès en dehors du dossier du projet ;
- fichiers sensibles (.env, clés, bases de données...) illisibles et non modifiables ;
- toute écriture demande une confirmation humaine, avec un diff.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
from pathlib import Path
from typing import Callable

MAX_OUTPUT_CHARS = 20_000
MAX_WRITE_BYTES = 200_000
MAX_WRITES_PER_SESSION = 20
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FILE_BYTES = 1_000_000

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode",
                "dist", "build", ".next", ".pytest_cache"}

SECRET_PATTERNS = [
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "*.jks", "*.keystore",
    "id_rsa*", "id_ed25519*", "*.sqlite", "*.sqlite3", "*.db", "*.sql", "*.dump",
    "*adminsdk*.json", "*service-account*.json", "*serviceaccount*.json",
    "secrets.*", "credentials*", ".netrc", ".npmrc", ".pypirc",
]
SECRET_EXCEPTIONS = {".env.example", ".env.sample", ".env.template"}

TOOL_NAMES = {"list_files", "read_file", "search_text", "write_file"}


class ToolError(Exception):
    """Erreur attendue, expliquée au modèle pour qu'il puisse s'adapter."""


def is_secret(path: Path) -> bool:
    name = path.name.lower()
    if name in SECRET_EXCEPTIONS:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_PATTERNS)


def is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\x00" in handle.read(2048)


def default_confirm(rel_path: str, preview: str) -> bool:
    print(f"\n--- Modification proposée : {rel_path} ---")
    print(preview)
    print("--- fin ---")
    answer = input("Appliquer cette modification ? [o/N] ")
    return answer.strip().lower() in {"o", "oui", "y", "yes"}


class Workspace:
    def __init__(self, root: str | Path, confirm: Callable[[str, str], bool] | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"Dossier de projet introuvable : {self.root}")
        self.confirm = confirm or default_confirm
        self.writes = 0

    # ---------- helpers ----------
    def _resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("Chemin invalide.")
        if ":" in rel or "\x00" in rel:
            # Bloque les lettres de lecteur et les flux alternatifs NTFS (fichier.txt:flux).
            raise ToolError("Chemin invalide : les caractères ':' et nuls sont interdits.")
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise ToolError("Chemin hors du projet : accès refusé.") from None
        return candidate

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _in_git_dir(self, path: Path) -> bool:
        return ".git" in path.relative_to(self.root).parts

    def _check_readable(self, path: Path) -> None:
        if not path.exists():
            raise ToolError(f"Fichier introuvable : {self._rel(path)}")
        if not path.is_file():
            raise ToolError(f"Ce n'est pas un fichier : {self._rel(path)}")
        if is_secret(path) or self._in_git_dir(path):
            raise ToolError("Accès refusé : fichier sensible ou interne à Git.")
        if is_binary(path):
            raise ToolError("Fichier binaire : lecture impossible.")

    # ---------- exécution ----------
    def run(self, name: str, args: dict) -> dict:
        """Exécute un outil. Ne lève jamais : renvoie {"result": ...} ou {"error": ...}."""
        if name not in TOOL_NAMES:
            return {"error": f"Outil inconnu : {name}"}
        if not isinstance(args, dict):
            args = {}
        try:
            text = getattr(self, f"tool_{name}")(**args)
        except ToolError as exc:
            return {"error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"error": f"Arguments invalides pour {name} : {exc}"}
        except (OSError, UnicodeError) as exc:
            return {"error": f"Erreur système : {exc}"}
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n[... sortie tronquée ...]"
        return {"result": text}

    # ---------- outils ----------
    def tool_list_files(self, path: str = ".") -> str:
        base = self._resolve(path)
        if not base.is_dir():
            raise ToolError(f"Ce n'est pas un dossier : {self._rel(base)}")
        entries = sorted(base.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries:
            if entry.is_dir():
                if entry.name in IGNORED_DIRS:
                    continue
                lines.append(f"{entry.name}/")
            elif not is_secret(entry):
                lines.append(f"{entry.name}  ({entry.stat().st_size} octets)")
        if not lines:
            return "(dossier vide)"
        extra = len(lines) - MAX_LIST_ENTRIES
        shown = lines[:MAX_LIST_ENTRIES]
        if extra > 0:
            shown.append(f"[... {extra} entrées de plus ...]")
        return "\n".join(shown)

    def tool_read_file(self, path: str, start_line: int = 1, max_lines: int = 300) -> str:
        target = self._resolve(path)
        self._check_readable(target)
        start_line = max(1, int(start_line))
        max_lines = max(1, min(1000, int(max_lines)))
        out, truncated_at = [], None
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if number < start_line:
                    continue
                if len(out) >= max_lines:
                    truncated_at = number
                    break
                out.append(f"{number:5}| {line.rstrip()[:500]}")
        if not out:
            return f"(aucune ligne à partir de la ligne {start_line})"
        text = "\n".join(out)
        if truncated_at:
            text += f"\n[... suite disponible : start_line={truncated_at} ...]"
        return text

    def tool_search_text(self, query: str, path: str = ".") -> str:
        query = str(query).strip()
        if len(query) < 2:
            raise ToolError("Requête trop courte (2 caractères minimum).")
        base = self._resolve(path)
        if not base.exists():
            raise ToolError(f"Chemin introuvable : {self._rel(base)}")
        needle = query.lower()
        files: list[Path] = []
        if base.is_file():
            files = [base]
        else:
            for folder, dirs, names in os.walk(base):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                files.extend(Path(folder) / name for name in sorted(names))
        results = []
        for file in files:
            try:
                if (is_secret(file) or file.stat().st_size > MAX_SEARCH_FILE_BYTES
                        or is_binary(file)):
                    continue
                with file.open("r", encoding="utf-8", errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        if needle in line.lower():
                            results.append(f"{self._rel(file)}:{number}: {line.strip()[:200]}")
                            if len(results) >= MAX_SEARCH_RESULTS:
                                results.append("[... résultats limités ...]")
                                return "\n".join(results)
            except OSError:
                continue
        return "\n".join(results) if results else "Aucun résultat."

    def tool_write_file(self, path: str, content: str) -> str:
        if not isinstance(content, str):
            raise ToolError("Le contenu doit être du texte.")
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ToolError("Contenu trop volumineux.")
        if self.writes >= MAX_WRITES_PER_SESSION:
            raise ToolError("Limite d'écritures atteinte pour cette session.")
        target = self._resolve(path)
        if target == self.root or target.is_dir():
            raise ToolError("Le chemin désigne un dossier, pas un fichier.")
        if is_secret(target) or self._in_git_dir(target):
            raise ToolError("Écriture refusée : fichier sensible ou interne à Git.")

        rel = self._rel(target)
        newline, old_text = "\n", None
        if target.exists():
            if is_binary(target):
                raise ToolError("Fichier binaire : modification impossible.")
            raw = target.read_bytes()
            if b"\r\n" in raw:
                newline = "\r\n"
            old_text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        new_text = content.replace("\r\n", "\n")

        if old_text is None:
            head = new_text.splitlines()[:40]
            preview = f"NOUVEAU FICHIER ({len(new_text.splitlines())} lignes)\n" + "\n".join(head)
        else:
            diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                             fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))
            if not diff:
                return "Aucun changement : le fichier est déjà identique."
            if len(diff) > 200:
                diff = diff[:200] + [f"[... diff tronqué ({len(diff) - 200} lignes de plus) ...]"]
            preview = "\n".join(diff)

        if not self.confirm(rel, preview):
            raise ToolError("Modification refusée par l'utilisateur. "
                            "Ne la réapplique pas sans nouvelle instruction.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(new_text.replace("\n", newline))
        self.writes += 1
        return f"OK : {rel} écrit ({len(new_text.splitlines())} lignes)."


TOOL_DECLARATIONS = [
    {
        "name": "list_files",
        "description": "Liste les fichiers et dossiers d'un dossier du projet.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Dossier relatif à la racine du projet. Défaut : '.'"}}},
    },
    {
        "name": "read_file",
        "description": "Lit un fichier texte du projet, avec numéros de ligne.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin relatif du fichier."},
            "start_line": {"type": "integer", "description": "Première ligne à lire (défaut 1)."},
            "max_lines": {"type": "integer", "description": "Nombre de lignes max (défaut 300, max 1000)."}},
            "required": ["path"]},
    },
    {
        "name": "search_text",
        "description": "Cherche un texte (insensible à la casse) dans les fichiers du projet.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Texte à chercher (2 caractères minimum)."},
            "path": {"type": "string", "description": "Dossier ou fichier où chercher. Défaut : '.'"}},
            "required": ["query"]},
    },
    {
        "name": "write_file",
        "description": ("Crée ou remplace un fichier avec le CONTENU COMPLET fourni. "
                        "L'utilisateur voit un diff et doit confirmer avant l'écriture."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Chemin relatif du fichier."},
            "content": {"type": "string", "description": "Contenu complet du fichier."}},
            "required": ["path", "content"]},
    },
]
