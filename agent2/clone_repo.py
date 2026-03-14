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
