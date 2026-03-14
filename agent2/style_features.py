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
        "indent_style": "tabs" if tabs_count > spaces_count else "spaces",
        "avg_function_length": (
            round(sum(function_lengths) / len(function_lengths), 2)
            if function_lengths
            else 0.0
        ),
    }

    logger.info("Features extracted: %s", features)
    return features
