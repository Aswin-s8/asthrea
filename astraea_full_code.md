# Astraea — Developer Code Fingerprinting System (Complete Source Code)

> **What this project does:** Given a GitHub username and a repository URL, this system verifies whether the repository was likely written by that developer. It does this by analyzing coding style (comment ratio, line length, indentation, function length) across the developer's other public repos and comparing it to the submitted repo, plus checking commit author metadata.

> **Tech stack:** Python, FastAPI, GitPython, Requests

> **How to run:**
> ```
> cd astraea
> pip install -r agent2/requirements.txt
> uvicorn agent2.fingerprint:app --host 0.0.0.0 --port 8000
> ```
> Then send a POST request to `/verify-developer` with `{"username": "...", "repo_url": "..."}`.

---

## Project Structure

```
astraea/
└── agent2/
    ├── __init__.py           # Makes agent2 a Python package
    ├── requirements.txt      # Dependencies
    ├── github_fetch.py       # Module 1: Fetch repos from GitHub API
    ├── clone_repo.py         # Module 2: Clone repos locally
    ├── style_features.py     # Module 3: Extract coding style features
    ├── similarity.py         # Module 4: Compute similarity score
    └── fingerprint.py        # Module 5: Orchestrator + FastAPI endpoint
```

---

## File 1: `requirements.txt`

```
fastapi
uvicorn[standard]
gitpython
requests
```

---

## File 2: `__init__.py`

```python
# Empty file — its presence makes `agent2/` a Python package so we can do
# relative imports like `from .github_fetch import fetch_repos`.
```

---

## File 3: `github_fetch.py` — Fetch Public Repos from GitHub API

```python
"""
github_fetch.py — Fetch all public repositories of a GitHub user.
"""

import logging
import requests

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"


def fetch_repos(username: str) -> list[str]:
    """
    Fetch all public, non-fork repositories for *username* via the GitHub API.
    Returns a list of HTTPS clone URLs.

    Raises:
        RuntimeError: If the GitHub API returns an error status.
    """
    logger.info("Fetching repositories for user '%s'", username)

    clone_urls: list[str] = []
    page = 1

    while True:
        url = f"{GITHUB_API_URL}/users/{username}/repos"
        params = {"per_page": 100, "page": page, "type": "owner"}

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"GitHub API request failed for user '{username}': {exc}"
            ) from exc

        repos = response.json()
        if not repos:
            break

        for repo in repos:
            if repo.get("fork"):
                continue
            clone_url = repo.get("clone_url")
            if clone_url:
                clone_urls.append(clone_url)

        page += 1

    logger.info("Found %d non-fork repositories for '%s'", len(clone_urls), username)
    return clone_urls
```

---

## File 4: `clone_repo.py` — Clone Repositories Locally

```python
"""
clone_repo.py — Clone repositories locally using GitPython.
"""

import logging
import os
import shutil
import git

logger = logging.getLogger(__name__)


def clone_repo(url: str, dest: str) -> str:
    """
    Clone a Git repository from *url* into *dest*.

    If *dest* already exists it is removed first to ensure a clean clone.
    Returns the absolute path to the cloned repository.

    Raises:
        RuntimeError: If the clone operation fails.
    """
    logger.info("Cloning %s → %s", url, dest)

    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)

        git.Repo.clone_from(url, dest, depth=1)  # shallow clone for speed
        logger.info("Successfully cloned %s", url)
        return os.path.abspath(dest)

    except git.GitCommandError as exc:
        raise RuntimeError(f"Could not clone repository '{url}': {exc}") from exc
```

---

## File 5: `style_features.py` — Extract Coding Style Features

```python
"""
style_features.py — Extract coding-style features from a Python repository.

Performance safeguards:
  • Only the first 50 .py files discovered are analysed.
  • The following directories are skipped while walking the repo:
    .git, node_modules, venv, dist, build, __pycache__
"""

import logging
import os

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "venv", "dist", "build", "__pycache__"}
MAX_FILES = 50


def _collect_py_files(repo_path: str) -> list[str]:
    """Return up to MAX_FILES Python file paths, skipping SKIP_DIRS."""
    py_files: list[str] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune unwanted directories in-place so os.walk skips them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
                if len(py_files) >= MAX_FILES:
                    return py_files

    return py_files


def extract_features(repo_path: str) -> dict:
    """
    Walk all (up to 50) Python files in *repo_path* and compute:

    - total_lines        – total number of lines across all files
    - comment_ratio      – ratio of comment lines to total lines
    - avg_line_length    – average number of characters per line
    - indent_style       – "spaces" | "tabs" (majority wins)
    - avg_function_length – average number of body lines inside ``def`` blocks

    Returns a feature dictionary.  If the repo contains no Python files,
    returns a dict with zeroed / default values.
    """
    logger.info("Extracting style features from %s", repo_path)

    py_files = _collect_py_files(repo_path)
    if not py_files:
        logger.warning("No Python files found in %s", repo_path)
        return {
            "total_lines": 0,
            "comment_ratio": 0.0,
            "avg_line_length": 0.0,
            "indent_style": "spaces",
            "avg_function_length": 0.0,
        }

    total_lines = 0
    comment_lines = 0
    total_line_length = 0
    tabs_count = 0
    spaces_count = 0

    # Function-length tracking
    function_lengths: list[int] = []
    in_function = False
    current_func_lines = 0
    func_indent = 0

    for fpath in py_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        for line in lines:
            total_lines += 1
            stripped = line.strip()
            total_line_length += len(line.rstrip("\n"))

            # Comment detection
            if stripped.startswith("#"):
                comment_lines += 1

            # Indentation detection
            if stripped and line[0] == "\t":
                tabs_count += 1
            elif stripped and line[0] == " ":
                spaces_count += 1

            # Function-length parsing
            if stripped.startswith("def "):
                # If we were already inside a function, save it
                if in_function:
                    function_lengths.append(current_func_lines)
                in_function = True
                current_func_lines = 0
                func_indent = len(line) - len(line.lstrip())
            elif in_function:
                if stripped == "":
                    # Blank lines inside the function body still count
                    current_func_lines += 1
                elif (len(line) - len(line.lstrip())) > func_indent:
                    current_func_lines += 1
                else:
                    # Dedented → function ended
                    function_lengths.append(current_func_lines)
                    in_function = False
                    current_func_lines = 0

        # End of file: close any open function
        if in_function:
            function_lengths.append(current_func_lines)
            in_function = False
            current_func_lines = 0

    features = {
        "total_lines": total_lines,
        "comment_ratio": round(comment_lines / total_lines, 4) if total_lines else 0.0,
        "avg_line_length": round(total_line_length / total_lines, 2) if total_lines else 0.0,
        "avg_function_length": (
            round(sum(function_lengths) / len(function_lengths), 2)
            if function_lengths
            else 0.0
        ),
        "indent_style": "tabs" if tabs_count > spaces_count else "spaces",
    }

    logger.info("Features extracted: %s", features)
    return features
```

---

## File 6: `similarity.py` — Compute Weighted Similarity Score

```python
"""
similarity.py — Compute a weighted similarity score between developer
repositories and a submitted patch repository.

Weights:
  comment_ratio       → 0.2
  avg_line_length     → 0.2
  indent_style        → 0.2
  avg_function_length → 0.4

The returned score is normalised to [0, 1].
"""

import logging

logger = logging.getLogger(__name__)

# Weights for each feature
WEIGHTS = {
    "comment_ratio": 0.2,
    "avg_line_length": 0.2,
    "indent_style": 0.2,
    "avg_function_length": 0.4,
}

# Maximum expected differences (used for normalisation)
MAX_DIFFS = {
    "comment_ratio": 1.0,        # ratio is 0-1
    "avg_line_length": 120.0,    # reasonable max diff in chars
    "avg_function_length": 100.0,  # reasonable max diff in lines
}


def _feature_similarity(dev_val, patch_val, key: str) -> float:
    """Return a 0-1 similarity for a single feature."""
    if key == "indent_style":
        return 1.0 if dev_val == patch_val else 0.0

    max_diff = MAX_DIFFS.get(key, 1.0)
    diff = abs(float(dev_val) - float(patch_val))
    return max(0.0, 1.0 - diff / max_diff)


def compute_similarity(dev_features_list: list[dict], patch_features: dict) -> float:
    """
    Average the developer's feature dicts, then compute a weighted similarity
    score against the patch repo's features.

    Returns a float in [0, 1].
    """
    if not dev_features_list:
        logger.warning("No developer features provided — returning 0.0")
        return 0.0

    # Average the developer features
    avg_dev: dict = {}
    for key in WEIGHTS:
        if key == "indent_style":
            # Majority vote
            styles = [f.get("indent_style", "spaces") for f in dev_features_list]
            avg_dev["indent_style"] = max(set(styles), key=styles.count)
        else:
            values = [f.get(key, 0.0) for f in dev_features_list]
            avg_dev[key] = sum(values) / len(values)

    logger.info("Averaged developer features: %s", avg_dev)

    # Weighted similarity
    score = 0.0
    for key, weight in WEIGHTS.items():
        sim = _feature_similarity(avg_dev.get(key, 0), patch_features.get(key, 0), key)
        score += weight * sim

    score = round(min(max(score, 0.0), 1.0), 4)
    logger.info("Computed similarity score: %s", score)
    return score
```

---

## File 7: `fingerprint.py` — Orchestrator + FastAPI Endpoint (Main Entry Point)

```python
"""
fingerprint.py — Orchestrator and FastAPI endpoint for the Astraea
Developer Code Fingerprinting system.
"""

import logging
import os
import tempfile

import git
from fastapi import FastAPI
from pydantic import BaseModel

from .github_fetch import fetch_repos
from .clone_repo import clone_repo
from .style_features import extract_features
from .similarity import compute_similarity

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Astraea — Developer Code Fingerprinting")


class VerifyRequest(BaseModel):
    username: str
    repo_url: str


# ---------------------------------------------------------------------------
# Commit verification helper
# ---------------------------------------------------------------------------

def _check_commit_author(repo_path: str, username: str) -> bool:
    """
    Inspect up to the latest 20 commits of the repository at *repo_path*.
    Return True if *username* (case-insensitive) appears in any commit's
    author name or author email.
    """
    logger.info("Checking commit metadata for username '%s'", username)
    try:
        repo = git.Repo(repo_path)
        lower_username = username.lower()
        for commit in list(repo.iter_commits(max_count=20)):
            author_name = (commit.author.name or "").lower()
            author_email = (commit.author.email or "").lower()
            if lower_username in author_name or lower_username in author_email:
                logger.info("Commit match found: %s <%s>", commit.author.name, commit.author.email)
                return True
    except Exception as exc:
        logger.error("Error reading commits: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------

def verify_developer(username: str, patch_repo_url: str) -> dict:
    """
    End-to-end developer fingerprint verification.

    Steps:
      1. Fetch the user's public repos.
      2. Clone the first 3 repos.
      3. Extract style features from each.
      4. Clone the patch repo.
      5. Extract its features.
      6. Compute similarity score.
      7. Check commit author metadata.
      8. Return a verification result dict.
    """

    tmp_dir = tempfile.mkdtemp(prefix="astraea_")

    # ------ Step 1: Fetch repos ------
    try:
        logger.info("Step 1/6 — Fetching repositories for '%s'", username)
        repo_urls = fetch_repos(username)
    except RuntimeError as exc:
        logger.error("Failed to fetch repos: %s", exc)
        return {"error": f"GitHub API error: {exc}"}

    if not repo_urls:
        logger.warning("User '%s' has no public repositories", username)
        return {"error": f"No public repositories found for user '{username}'"}

    # ------ Step 2: Clone first 3 repos ------
    logger.info("Step 2/6 — Cloning up to 3 developer repos")
    dev_paths: list[str] = []
    for i, url in enumerate(repo_urls[:3]):
        dest = os.path.join(tmp_dir, f"dev_repo_{i}")
        try:
            path = clone_repo(url, dest)
            dev_paths.append(path)
        except RuntimeError as exc:
            logger.warning("Skipping repo %s: %s", url, exc)

    if not dev_paths:
        return {"error": "Could not clone any developer repositories"}

    # ------ Step 3: Extract features ------
    logger.info("Step 3/6 — Extracting features from developer repos")
    dev_features = [extract_features(p) for p in dev_paths]

    # ------ Step 4: Clone patch repo ------
    logger.info("Step 4/6 — Cloning patch repo")
    patch_dest = os.path.join(tmp_dir, "patch_repo")
    try:
        patch_path = clone_repo(patch_repo_url, patch_dest)
    except RuntimeError as exc:
        logger.error("Failed to clone patch repo: %s", exc)
        return {"error": f"Could not clone repository: {exc}"}

    # ------ Step 5: Extract patch features ------
    logger.info("Step 5/6 — Extracting features from patch repo")
    patch_features = extract_features(patch_path)

    # ------ Step 6: Compute similarity ------
    logger.info("Step 6/6 — Computing similarity & checking commits")
    style_similarity = compute_similarity(dev_features, patch_features)
    commit_match = _check_commit_author(patch_path, username)

    # Ownership score: blend of style similarity and commit signal
    ownership_score = round(
        style_similarity * 0.7 + (0.3 if commit_match else 0.0), 2
    )

    verified = ownership_score > 0.6 and commit_match

    result = {
        "ownership_score": ownership_score,
        "style_similarity": round(style_similarity, 2),
        "commit_match": commit_match,
        "verified": verified,
    }

    logger.info("Verification result: %s", result)
    return result


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------

@app.post("/verify-developer")
async def verify_developer_endpoint(req: VerifyRequest):
    """POST /verify-developer — run the full fingerprint verification."""
    return verify_developer(req.username, req.repo_url)
```

---

## How the System Works (Data Flow)

```
POST /verify-developer { username, repo_url }
        │
        ▼
  ┌─────────────────┐
  │  github_fetch.py │  ──►  GitHub API: get user's public repos
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  clone_repo.py   │  ──►  Shallow-clone 3 dev repos + the submitted repo
  └────────┬────────┘
           ▼
  ┌──────────────────┐
  │ style_features.py │  ──►  Extract: comment_ratio, avg_line_length,
  └────────┬─────────┘       indent_style, avg_function_length
           ▼
  ┌─────────────────┐
  │  similarity.py   │  ──►  Weighted comparison (0.2/0.2/0.2/0.4)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ fingerprint.py   │  ──►  Blend similarity + commit check → verified T/F
  └─────────────────┘
```

## Example Output

```json
{
    "ownership_score": 0.96,
    "style_similarity": 0.94,
    "commit_match": true,
    "verified": true
}
```

- **ownership_score**: Final blended score (style_similarity × 0.7 + 0.3 if commits match)
- **style_similarity**: How similar the coding style is (0 to 1)
- **commit_match**: Whether the username appears in commit author metadata
- **verified**: `true` only if ownership_score > 0.6 AND commit_match is true
