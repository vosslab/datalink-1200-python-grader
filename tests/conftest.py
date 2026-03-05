"""Pytest configuration: add repo root to sys.path for module imports."""

# Standard Library
import os
import sys

# add tests/ directory first so git_file_utils is importable
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
	sys.path.insert(0, TESTS_DIR)

# local repo modules
import git_file_utils

# add repo root to sys.path so tests can import omr_utils, grade_answers, etc.
REPO_ROOT = git_file_utils.get_repo_root()
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)
