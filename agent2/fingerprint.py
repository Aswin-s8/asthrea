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

from dotenv import load_dotenv

from .github_fetch import fetch_repos
from .clone_repo import clone_repo
from .style_features import extract_features
from .similarity import compute_similarity
from .llm_analysis import analyze_semantic_style

load_dotenv() # Load keys from .env

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
    groq_api_key: str = None


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
      7. Perform semantic LLM analysis (if key available).
      8. Check commit author metadata.
      9. Return a verification result dict.
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
    logger.info("Step 6/7 — Computing similarity & checking commits")
    style_similarity = compute_similarity(dev_features, patch_features)
    commit_match = _check_commit_author(patch_path, username)

    # ------ Step 7: LLM Semantic Analysis ------
    logger.info("Step 7/7 — Semantic LLM analysis")
    api_key = os.getenv("GROQ_API_KEY")
    semantic_result = analyze_semantic_style(dev_paths, patch_path, api_key)
    semantic_score = semantic_result.get("score", 0.0)

    # Ownership score: blend of style similarity, commit signal, and LLM semantic analysis
    # Weights: Style (0.4) + Commit (0.2) + LLM Semantic (0.4)
    ownership_score = round(
        style_similarity * 0.4 + 
        (0.2 if commit_match else 0.0) +
        semantic_score * 0.4, 2
    )

    # Verified if ownership score is reasonable and we have some positive signal
    verified = ownership_score > 0.6 and (commit_match or semantic_score > 0.7)

    result = {
        "ownership_score": ownership_score,
        "style_similarity": round(style_similarity, 2),
        "semantic_similarity": round(semantic_score, 2),
        "semantic_explanation": semantic_result.get("explanation"),
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
