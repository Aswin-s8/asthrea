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
