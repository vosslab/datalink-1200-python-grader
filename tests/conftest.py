"""Pytest configuration: add repo root to sys.path for module imports."""

# Standard Library
import os
import sys

# add repo root to sys.path so tests can import omr_utils, grade_answers, etc.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

# add tests/ directory so git_file_utils is importable
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
if TESTS_DIR not in sys.path:
	sys.path.insert(0, TESTS_DIR)
