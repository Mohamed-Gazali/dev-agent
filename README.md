# DevAgent — assistant de développement en ligne de commande

![tests](https://github.com/Mohamed-Gazali/dev-agent/actions/workflows/tests.yml/badge.svg)

Un agent de code minimal, écrit à des fins d'apprentissage : **un modèle Gemini (via API) + 4 outils + une boucle**.
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
copy .env.example .env      # puis renseigne GEMINI_API_KEY (clé gratuite : Google AI Studio)
```

## Utilisation

```powershell
python agent.py --project C:\chemin\vers\un-projet-de-test --verbose
```

Options : `--model` (nom du modèle), `--max-steps` (étapes max par demande, 12 par défaut), `--verbose` (tokens et requêtes).
Commandes dans l'agent : `/reset` (oublier la conversation), `/quit`.

## Tests (sans réseau, sans clé)

```powershell
python -m unittest -v
```

Les tests couvrent les outils (sorties de chemin, fichiers sensibles, confirmation, fins de ligne), la boucle d'agent
(avec une API simulée), la gestion des erreurs et des quotas.

## Quotas du niveau gratuit

Chaque modèle gratuit a ses propres limites (certains comptes : 5 requêtes/minute, 20 requêtes/jour — à vérifier sur
https://aistudio.google.com/rate-limit). **Chaque étape de l'agent est une requête API.** Pour économiser : groupe tes
consignes, tape `/reset` entre deux sujets, et relance avec un autre modèle quand le quota du jour est atteint
(`--model ...`).

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

- Un seul fournisseur d'API pour l'instant (Gemini).
- Avec un `--max-steps` très bas, la réponse finale peut manquer ; une modification confirmée reste appliquée.
- `write_file` remplace le fichier en entier : sur un gros fichier, le modèle peut omettre du contenu — relis le diff.
- Testé sur de petits projets d'exercice uniquement, pas sur de gros dépôts.
- Les noms de modèles changent vite : ajuste `GEMINI_MODEL` dans `.env` si besoin.

## Structure

- `llm.py` : appel HTTP à l'API, erreurs, quotas, nouvelles tentatives
- `tools.py` : les outils et toutes les règles de sécurité
- `agent.py` : boucle d'agent et interface en ligne de commande
- `test_agent.py` : tests unitaires

## Feuille de route

1. Plusieurs fournisseurs d'API (répartition des quotas)
2. Outil `run_tests` avec liste blanche de commandes
3. Branche Git + Pull Request automatiques

## Licence

MIT — voir [LICENSE](LICENSE).
