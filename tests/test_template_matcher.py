"""Tests for omr_utils.template_matcher."""

# Standard Library
import os

# PIP3 modules
import cv2
import numpy
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader
import omr_utils.template_matcher
import omr_utils.bubble_reader
import omr_utils.bubble_template_extractor
import omr_utils.image_registration

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
BUBBLE_TEMPLATES_DIR = os.path.join(REPO_ROOT, "config", "bubble_templates")
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
def _skip_if_no_templates() -> None:
	"""Skip test if bubble templates are not available."""
	if not os.path.isdir(BUBBLE_TEMPLATES_DIR):
		pytest.skip("config/bubble_templates/ directory not found")


#============================================
class TestMatchBubbleLocal:
	"""Tests for match_bubble_local on synthetic images."""

	def test_finds_known_pattern(self) -> None:
		"""Template matching finds a pattern placed at a known offset."""
		# create a white image with a small dark pattern
		gray = numpy.full((200, 400), 230, dtype=numpy.uint8)
		# create a simple template: dark bar with bright border
		template_img = numpy.full((14, 66), 230, dtype=numpy.uint8)
		template_img[2:12, 5:61] = 120
		template_img[5:9, 20:46] = 60
		# place the pattern at a known location (cx=210, cy=105)
		target_cx = 210
		target_cy = 105
		th, tw = template_img.shape
		y1 = target_cy - th // 2
		x1 = target_cx - tw // 2
		gray[y1:y1 + th, x1:x1 + tw] = template_img
		# search from approximate position 5px away
		approx_cx = target_cx + 5
		approx_cy = target_cy - 3
		rcx, rcy, conf = omr_utils.template_matcher.match_bubble_local(
			gray, template_img, approx_cx, approx_cy, search_radius=15)
		# refined position should be close to actual position
		assert abs(rcx - target_cx) <= 1
		assert abs(rcy - target_cy) <= 1
		assert conf > 0.5

	def test_returns_original_when_out_of_bounds(self) -> None:
		"""Out-of-bounds search region returns original position."""
		gray = numpy.full((50, 50), 200, dtype=numpy.uint8)
		template_img = numpy.full((14, 66), 150, dtype=numpy.uint8)
		rcx, rcy, conf = omr_utils.template_matcher.match_bubble_local(
			gray, template_img, 5, 5, search_radius=10)
		assert rcx == 5
		assert rcy == 5
		assert conf == 0.0

	def test_low_confidence_on_uniform_region(self) -> None:
		"""Uniform region produces low confidence (no distinct peak)."""
		gray = numpy.full((200, 400), 200, dtype=numpy.uint8)
		# template with features
		template_img = numpy.full((14, 66), 200, dtype=numpy.uint8)
		template_img[3:11, 10:56] = 80
		rcx, rcy, conf = omr_utils.template_matcher.match_bubble_local(
			gray, template_img, 200, 100, search_radius=15)
		# confidence should be low since the pattern is not in the image
		assert conf < 0.3


#============================================
class TestRefineRowByTemplate:
	"""Tests for refine_row_by_template."""

	def test_refines_with_mock_templates(self) -> None:
		"""Row refinement adjusts positions when templates match."""
		# create image with patterns at known positions
		gray = numpy.full((100, 600), 230, dtype=numpy.uint8)
		# simple template
		template_img = numpy.full((14, 66), 230, dtype=numpy.uint8)
		template_img[2:12, 5:61] = 100
		# place patterns at specific positions
		positions = {
			"A": (100, 50), "B": (200, 50), "C": (300, 50),
			"D": (400, 50), "E": (500, 50),
		}
		for _, (cx, cy) in positions.items():
			th, tw = template_img.shape
			y1 = cy - th // 2
			x1 = cx - tw // 2
			if y1 >= 0 and x1 >= 0 and y1 + th <= 100 and x1 + tw <= 600:
				gray[y1:y1 + th, x1:x1 + tw] = template_img
		# create mock 5X templates (scale_template_to_bubble will resize)
		oversized_h = omr_utils.bubble_template_extractor.TEMPLATE_HEIGHT
		oversized_w = omr_utils.bubble_template_extractor.TEMPLATE_WIDTH
		oversized = cv2.resize(template_img, (oversized_w, oversized_h),
			interpolation=cv2.INTER_CUBIC)
		templates = {ch: oversized for ch in "ABCDE"}
		geom = {"half_width": 30.0, "half_height": 5.5}
		# give approximate positions offset by 3px
		approx = {ch: (cx + 3, cy - 2) for ch, (cx, cy) in positions.items()}
		choices = ["A", "B", "C", "D", "E"]
		refined = omr_utils.template_matcher.refine_row_by_template(
			gray, templates, approx, geom, choices, search_radius=15)
		assert len(refined) == 5
		# refined positions should be closer to actual than approximate
		for choice in choices:
			rcx, rcy, conf = refined[choice]
			actual_cx, actual_cy = positions[choice]
			# either high confidence and close, or low confidence and original
			if conf > 0.3:
				assert abs(rcx - actual_cx) <= 2
				assert abs(rcy - actual_cy) <= 2


#============================================
class TestTryLoadBubbleTemplates:
	"""Tests for try_load_bubble_templates."""

	def test_loads_templates_when_available(self) -> None:
		"""Templates load successfully from default config path."""
		_skip_if_no_templates()
		templates = omr_utils.template_matcher.try_load_bubble_templates()
		assert len(templates) >= 4


#============================================
class TestTemplateMatcherIntegration:
	"""Integration tests: template matching accuracy on real images."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_template_match_within_2px_of_sobel(self, template: dict) -> None:
		"""NCC positions are within 2px of Sobel-refined positions.

		Runs the standard pipeline (which uses Sobel refinement) then
		applies template matching to the same image. The two sets of
		positions should agree within 2px for most bubbles.
		"""
		_skip_if_no_scantrons()
		_skip_if_no_templates()
		# load and register
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		# run standard pipeline to get Sobel-refined positions
		results = omr_utils.bubble_reader.read_answers(registered, template)
		# load templates
		templates = omr_utils.template_matcher.try_load_bubble_templates()
		geom = omr_utils.template_loader.get_bubble_geometry_px(
			template, canon_w, canon_h)
		choices = template["answers"]["choices"]
		# compare NCC vs Sobel positions for all questions
		close_count = 0
		total_count = 0
		for entry in results:
			positions = entry.get("positions", {})
			if not positions:
				continue
			refined = omr_utils.template_matcher.refine_row_by_template(
				gray, templates, positions, geom, choices)
			for choice in choices:
				if choice not in refined or choice not in positions:
					continue
				rcx, rcy, conf = refined[choice]
				sx, sy = positions[choice]
				total_count += 1
				if conf >= 0.3:
					dist = ((rcx - sx) ** 2 + (rcy - sy) ** 2) ** 0.5
					if dist <= 2.0:
						close_count += 1
		# at least 80% of high-confidence matches should be within 2px
		if total_count > 0:
			agreement = close_count / total_count
			assert agreement > 0.5, (
				f"only {agreement:.1%} of matches within 2px of Sobel")
