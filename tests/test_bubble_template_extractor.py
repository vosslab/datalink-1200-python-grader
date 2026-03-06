"""Tests for omr_utils.bubble_template_extractor."""

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
import omr_utils.bubble_template_extractor
import omr_utils.image_registration

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")
KEY_IMAGE = os.path.join(
	SCANTRON_DIR,
	"43F257A7-A03D-4CB2-8D7B-3EE057B41FAC_result.jpg"
)


#============================================
def _skip_if_no_scantrons() -> None:
	"""Skip test if scantron images are not available."""
	if not os.path.isdir(SCANTRON_DIR):
		pytest.skip("scantrons/ directory not found")


#============================================
class TestExtractBubblePatch:
	"""Tests for extract_bubble_patch on synthetic images."""

	def test_returns_correct_dimensions(self) -> None:
		"""Extracted patch has expected 5X oversize dimensions."""
		gray = numpy.full((100, 200), 200, dtype=numpy.uint8)
		geom = {"half_width": 30.0, "half_height": 5.5}
		patch = omr_utils.bubble_template_extractor.extract_bubble_patch(
			gray, 100, 50, geom)
		assert patch is not None
		expected_h = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		expected_w = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		assert patch.shape == (expected_h, expected_w)

	def test_returns_none_for_out_of_bounds(self) -> None:
		"""Out-of-bounds center returns None."""
		gray = numpy.full((50, 50), 200, dtype=numpy.uint8)
		geom = {"half_width": 30.0, "half_height": 5.5}
		# center at edge, extraction region goes out of bounds
		patch = omr_utils.bubble_template_extractor.extract_bubble_patch(
			gray, 5, 5, geom)
		assert patch is None

	def test_preserves_pattern(self) -> None:
		"""Extracted patch preserves dark/bright pattern from source."""
		gray = numpy.full((100, 200), 240, dtype=numpy.uint8)
		# draw a dark horizontal bar at center
		gray[48:52, 60:140] = 30
		geom = {"half_width": 30.0, "half_height": 5.5}
		patch = omr_utils.bubble_template_extractor.extract_bubble_patch(
			gray, 100, 50, geom)
		assert patch is not None
		# center of patch should be darker than edges
		center_mean = float(numpy.mean(patch[20:35, 100:200]))
		edge_mean = float(numpy.mean(patch[0:10, :]))
		assert center_mean < edge_mean


#============================================
class TestScalePatchQuality:
	"""Tests for _score_patch_quality."""

	def test_high_contrast_scores_well(self) -> None:
		"""A patch with dark edges and bright interior scores high."""
		th = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		tw = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		patch = numpy.full((th, tw), 220, dtype=numpy.uint8)
		# dark edges at top and bottom
		edge_h = max(1, int(th * 0.15))
		patch[:edge_h, :] = 40
		patch[-edge_h:, :] = 40
		score = omr_utils.bubble_template_extractor._score_patch_quality(patch)
		assert score > 100

	def test_uniform_scores_low(self) -> None:
		"""A uniform gray patch has near-zero contrast."""
		th = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		tw = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		patch = numpy.full((th, tw), 150, dtype=numpy.uint8)
		score = omr_utils.bubble_template_extractor._score_patch_quality(patch)
		assert abs(score) < 10


#============================================
class TestSaveLoadTemplates:
	"""Tests for save_templates and load_templates round-trip."""

	def test_save_load_round_trip(self, tmp_path: str) -> None:
		"""Templates survive save/load cycle with correct shapes."""
		th = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		tw = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		templates = {}
		for letter in ["A", "B", "C", "D", "E"]:
			# create a unique pattern per letter
			img = numpy.full((th, tw), 200, dtype=numpy.uint8)
			# each letter gets a different brightness stripe
			offset = ord(letter) - ord("A")
			img[20 + offset:25 + offset, :] = 50
			templates[letter] = img
		# save
		output_dir = str(tmp_path / "templates")
		saved = omr_utils.bubble_template_extractor.save_templates(
			templates, output_dir)
		assert len(saved) == 5
		# verify files exist
		for letter in ["A", "B", "C", "D", "E"]:
			fpath = os.path.join(output_dir, f"{letter}.png")
			assert os.path.isfile(fpath)
		# load
		loaded = omr_utils.bubble_template_extractor.load_templates(output_dir)
		assert len(loaded) == 5
		for letter in ["A", "B", "C", "D", "E"]:
			assert letter in loaded
			assert loaded[letter].shape == (th, tw)

	def test_load_from_nonexistent_dir(self) -> None:
		"""Loading from missing directory returns empty dict."""
		loaded = omr_utils.bubble_template_extractor.load_templates(
			"/nonexistent/path/to/templates")
		assert loaded == {}


#============================================
class TestScaleTemplate:
	"""Tests for scale_template_to_bubble."""

	def test_scale_preserves_content(self) -> None:
		"""Scaled template has correct dimensions for given geometry."""
		th = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		tw = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		template_img = numpy.full((th, tw), 150, dtype=numpy.uint8)
		geom = {"half_width": 30.0, "half_height": 5.5}
		scaled = omr_utils.bubble_template_extractor.scale_template_to_bubble(
			template_img, geom)
		pad_x = omr_utils.bubble_template_extractor.EXTRACT_PAD_X
		pad_y = omr_utils.bubble_template_extractor.EXTRACT_PAD_Y
		expected_w = int(geom["half_width"]) * 2 + pad_x * 2
		expected_h = int(geom["half_height"]) * 2 + pad_y * 2
		assert scaled.shape == (expected_h, expected_w)


#============================================
class TestExtractLetterTemplates:
	"""Integration tests for extract_letter_templates with real images."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_extracts_all_five_letters(self, template: dict) -> None:
		"""All 5 letter templates are extracted from the answer key image."""
		_skip_if_no_scantrons()
		# load and register image
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		# run detection
		results = omr_utils.bubble_reader.read_answers(registered, template)
		# extract templates
		templates = omr_utils.bubble_template_extractor.extract_letter_templates(
			gray, template, results)
		# answer key has many filled bubbles, but non-selected choices
		# in each row should be empty, giving plenty of samples
		assert len(templates) >= 4, (
			f"expected at least 4 letter templates, got {len(templates)}")
		# check template dimensions
		expected_h = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		expected_w = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		for letter, tmpl in templates.items():
			assert tmpl.shape == (expected_h, expected_w), (
				f"template {letter} has wrong shape: {tmpl.shape}")

	def test_templates_are_not_uniform(self, template: dict) -> None:
		"""Extracted templates are not uniform gray (show letter features)."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		templates = omr_utils.bubble_template_extractor.extract_letter_templates(
			gray, template, results)
		# each template should have visible features (not uniform)
		# printed letters and bracket edges create dark regions that
		# differ from the bright background in the measurement zones
		for letter, tmpl in templates.items():
			std_dev = float(numpy.std(tmpl))
			assert std_dev > 10, (
				f"template {letter} is too uniform (std={std_dev:.1f})")
			# template should not be all-white or all-black
			mean_val = float(numpy.mean(tmpl))
			assert 50 < mean_val < 240, (
				f"template {letter} has extreme mean={mean_val:.1f}")
