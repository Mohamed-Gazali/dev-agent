# DevAgent — assistant de développement en ligne de commande

![tests](https://github.com/Mohamed-Gazali/dev-agent/actions/workflows/tests.yml/badge.svg)

Un agent de code minimal, écrit à des fins d'apprentissage : **un modèle de langage (via API) + 4 outils + une boucle**.
Le fournisseur d'API est interchangeable : Google Gemini, Groq, ou toute API « compatible OpenAI ».
Le modèle explore un dossier de projet, propose des modifications, et **rien n'est écrit sans ta confirmation**.

## Ce que fait l'agent

| Outil | Rôle | Garde-fous (appliqués dans le code, pas dans le prompt) |
|---|---|---|
| `list_files` | Lister un dossier | Masque `.git`, `node_modules`, `.venv` et les fichiers sensibles |
| `read_file` | Lire un fichier | Refuse `.env`, clés, bases de données, binaires, tout chemin hors-projet |
| `search_text` | Chercher du texte | Mêmes règles |
| `write_file` | Créer / modifier un fichier | **Diff + confirmation obligatoire** ; jamais `.git`, ni secrets, ni hors-projet |

L'agent **ne peut pas** exécuter de commandes (ni tests, ni `pip`).

## Exemple de session

```text
vous > Lis test_inventaire.py, puis corrige inventaire.py pour que le test test_remove_trop_leve_une_erreur passe.
  → read_file(path='test_inventaire.py')
  → read_file(path='inventaire.py')
  → write_file(path='inventaire.py')

--- Modification proposée : inventaire.py ---
     def remove(self, name, qty):
-        self.items[name] = self.items.get(name, 0) - qty
+        current_qty = self.items.get(name, 0)
+        if qty > current_qty:
+            raise ValueError("Quantité insuffisante en stock")
+        self.items[name] = current_qty - qty
--- fin ---
Appliquer cette modification ? [o/N] o
```

## Installation (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # puis renseigne la clé du fournisseur choisi (voir ci-dessous)
```

## Utilisation

```powershell
python agent.py --project C:\chemin\vers\un-projet-de-test --verbose                 # Gemini (défaut)
python agent.py --project C:\chemin\vers\un-projet-de-test --provider groq --verbose  # Groq
```

Options : `--provider` (`gemini`, `groq`, `custom`), `--model` (nom du modèle), `--base-url` (fournisseur `custom`),
`--max-steps` (étapes max par demande, 12 par défaut), `--max-output-chars` (taille max d'un résultat d'outil envoyé
au modèle), `--verbose` (tokens et requêtes).
Commandes dans l'agent : `/reset` (oublier la conversation), `/quit`.

## Tests (sans réseau, sans clé)

```powershell
python -m unittest -v
```

Les tests couvrent les outils (sorties de chemin, fichiers sensibles, confirmation, fins de ligne), la boucle d'agent
avec une API simulée (formats Gemini et OpenAI), le choix du fournisseur, la gestion des erreurs et des quotas.

## Fournisseurs d'API

| Fournisseur | Clé | Variables `.env` | Remarque |
|---|---|---|---|
| `gemini` (défaut) | [Google AI Studio](https://aistudio.google.com/apikey) | `GEMINI_API_KEY`, `GEMINI_MODEL` | Limites gratuites par modèle, parfois très basses (ex. 20 requêtes/jour) |
| `groq` | [Groq Console](https://console.groq.com/keys) | `GROQ_API_KEY`, `GROQ_MODEL` | Limites gratuites exprimées en requêtes **et en tokens** (voir ci-dessous) |
| `custom` | selon le service | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Toute API `/chat/completions` compatible OpenAI (non testée ici avec de vrais services) |

Les noms de modèles et les limites changent vite : vérifie-les dans la console du fournisseur.
Exemple `custom` : `python agent.py --project ... --provider custom --base-url https://.../v1 --model nom-du-modele`.

### Quotas du niveau gratuit

**Chaque étape de l'agent est une requête API.** Les limites s'appliquent par modèle (et, chez Groq, par organisation) :
requêtes par minute, par jour, et souvent tokens par minute / par jour. Quand une limite est atteinte :
- l'agent s'arrête avec un message clair (pas de nouvelles tentatives inutiles sur un quota journalier) ;
- relance avec `--model` ou `--provider` différent (une nouvelle conversation, car l'historique est propre à chaque format d'API) ;
- baisse `--max-output-chars` si le fournisseur limite les tokens par minute ;
- tape `/reset` entre deux sujets et groupe tes consignes.

## Sécurité : ce qui est garanti, et ce qui ne l'est pas

**Garanti par le code** (testé) : pas d'accès hors du dossier du projet, pas de lecture ni d'écriture des fichiers
sensibles, pas d'écriture dans `.git`, confirmation humaine avant toute écriture.

**Non garanti** :
- Le contenu des fichiers lus est **envoyé à l'API de Google**. Sur le niveau gratuit, Google peut utiliser les requêtes pour
  améliorer ses produits : ne l'utilise pas sur du code confidentiel ni sur des données personnelles.
- Une clé écrite en dur *dans un fichier de code* n'est pas détectée.
- Une **injection de prompt** (instructions cachées dans un fichier) peut tromper le modèle : c'est la confirmation avant
  écriture qui te protège. Relis chaque diff.
- Un modèle peut proposer un changement faux ou incomplet : des tests verts ne prouvent pas que le code est correct.

Travaille toujours sur une **branche Git** pour pouvoir annuler (`git diff`, `git restore`).

## Limites connues

- Pas de bascule automatique entre fournisseurs : l'historique est stocké dans le format de chaque API, il faut relancer.
- Le comportement des appels d'outils varie selon le modèle : un petit modèle peut mal former un appel (l'erreur est alors renvoyée au modèle).
- Avec un `--max-steps` très bas, la réponse finale peut manquer ; une modification confirmée reste appliquée.
- `write_file` remplace le fichier en entier : sur un gros fichier, le modèle peut omettre du contenu — relis le diff.
- Testé sur de petits projets d'exercice uniquement, pas sur de gros dépôts.
- Les noms de modèles changent vite : ajuste `GEMINI_MODEL` dans `.env` si besoin.

## Structure

- `llm.py` : clients d'API (Gemini, compatible OpenAI) derrière une interface commune ; erreurs, quotas, nouvelles tentatives
- `providers.py` : choix du fournisseur et réglages par défaut
- `tools.py` : les outils et toutes les règles de sécurité
- `agent.py` : boucle d'agent et interface en ligne de commande
- `test_agent.py` : tests unitaires

## Feuille de route

1. ~~Plusieurs fournisseurs d'API~~ (v0.2)
2. Historique neutre pour basculer de fournisseur en pleine conversation
3. Outil `run_tests` avec liste blanche de commandes
4. Branche Git + Pull Request automatiques

## Licence

MIT — voir [LICENSE](LICENSE).
