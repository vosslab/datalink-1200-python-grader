"""Tests for omr_utils.bubble_reader."""

# Standard Library
import os

# PIP3 modules
import cv2
import numpy
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader
import omr_utils.bubble_reader
import omr_utils.image_registration

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")
BOTELLO = os.path.join(SCANTRON_DIR, "Botello_2.24.26.jpg")


#============================================
def _skip_if_no_scantrons() -> None:
	"""Skip test if scantron images are not available."""
	if not os.path.isdir(SCANTRON_DIR):
		pytest.skip("scantrons/ directory not found")


#============================================
class TestScoreBubbleFast:
	"""Tests for score_bubble_fast function."""

	def test_white_area_low_score(self) -> None:
		"""A white region should have a low or zero score."""
		# all white image
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		score = omr_utils.bubble_reader.score_bubble_fast(gray, 50, 50, 10)
		assert score <= 0.01

	def test_black_rect_high_score(self) -> None:
		"""A black rectangle on white background should score high."""
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		# draw a filled black rectangle matching bubble box dimensions
		hw = omr_utils.bubble_reader.BUBBLE_HALF_WIDTH
		hh = omr_utils.bubble_reader.BUBBLE_HALF_HEIGHT
		cv2.rectangle(gray, (50 - hw, 50 - hh), (50 + hw, 50 + hh), 0, -1)
		score = omr_utils.bubble_reader.score_bubble_fast(gray, 50, 50, 10)
		assert score > 0.3

	def test_out_of_bounds_negative(self) -> None:
		"""Out-of-bounds coordinates return -1."""
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		score = omr_utils.bubble_reader.score_bubble_fast(gray, -10, -10, 10)
		assert score == -1.0


#============================================
class TestReadAnswers:
	"""Tests for read_answers with real scantron images."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	@pytest.fixture()
	def botello_registered(self, template: dict) -> numpy.ndarray:
		"""Register and return the Botello image."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(BOTELLO)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		return registered

	def test_returns_100_results(self, template: dict,
		botello_registered: numpy.ndarray) -> None:
		"""read_answers returns exactly 100 question results."""
		results = omr_utils.bubble_reader.read_answers(
			botello_registered, template)
		assert len(results) == 100

	def test_result_keys(self, template: dict,
		botello_registered: numpy.ndarray) -> None:
		"""Each result has required keys."""
		results = omr_utils.bubble_reader.read_answers(
			botello_registered, template)
		first = results[0]
		assert "question" in first
		assert "answer" in first
		assert "scores" in first
		assert "flags" in first

	def test_q28_is_e(self, template: dict,
		botello_registered: numpy.ndarray) -> None:
		"""Question 28 answer should be E (known from all forms)."""
		results = omr_utils.bubble_reader.read_answers(
			botello_registered, template)
		# find Q28
		q28 = [r for r in results if r["question"] == 28][0]
		assert q28["answer"] == "E"

	def test_most_questions_answered(self, template: dict,
		botello_registered: numpy.ndarray) -> None:
		"""At least 50 questions should have answers (non-blank)."""
		results = omr_utils.bubble_reader.read_answers(
			botello_registered, template)
		answered = sum(1 for r in results if r["answer"])
		assert answered >= 50

	def test_valid_choices_only(self, template: dict,
		botello_registered: numpy.ndarray) -> None:
		"""All detected answers should be valid choices (A-E) or blank."""
		results = omr_utils.bubble_reader.read_answers(
			botello_registered, template)
		valid = {"A", "B", "C", "D", "E", ""}
		for r in results:
			assert r["answer"] in valid
