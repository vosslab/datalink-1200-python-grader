"""Tests for omr_utils.bubble_reader."""

# Standard Library
import os

# PIP3 modules
import numpy
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader
import omr_utils.bubble_reader
import omr_utils.debug_drawing
import omr_utils.image_registration

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")
# answer key image (not student data, safe to hardcode)
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
def _discover_student_images() -> list:
	"""Discover student scantron images (all except answer key).

	Returns:
		list of absolute paths to student scantron images
	"""
	key_basename = os.path.basename(KEY_IMAGE)
	image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
	student_images = []
	for fname in sorted(os.listdir(SCANTRON_DIR)):
		if fname == key_basename:
			continue
		if fname.lower().endswith(image_extensions):
			student_images.append(os.path.join(SCANTRON_DIR, fname))
	return student_images


#============================================
def _make_default_geom() -> dict:
	"""Return default geometry dict for synthetic tests at canonical size.

	Casts float values to int for use in array slicing within tests.
	"""
	geom = omr_utils.bubble_reader._default_geom()
	int_geom = {k: int(v) for k, v in geom.items()}
	return int_geom


#============================================
class TestScoreBubbleFast:
	"""Tests for score_bubble_fast function."""

	def test_white_area_low_score(self) -> None:
		"""A white region should yield a low score.

		With bracket-edge scoring, both bracket edges and measurement
		zone are white, so score should be near zero.
		"""
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		score = omr_utils.bubble_reader.score_bubble_fast(
			gray, 50, 50, 10)
		assert score <= 0.05

	def test_black_rect_high_score(self) -> None:
		"""A black rectangle with dark bracket edges should score high.

		Creates a synthetic image with dark bracket edges and a dark
		measurement zone, simulating a filled bubble on a real form.
		"""
		geom = _make_default_geom()
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		hw = geom["half_width"]
		hh = geom["half_height"]
		beh = geom["bracket_edge_height"]
		cx, cy = 50, 50
		# draw dark bracket edges (top and bottom strips)
		gray[cy - hh:cy - hh + beh, cx - hw:cx + hw] = 40
		gray[cy + hh - beh:cy + hh, cx - hw:cx + hw] = 40
		# draw dark fill in measurement zone (simulating pencil marks)
		mi_v = geom["measurement_inset_v"]
		mz_y1 = cy - hh + mi_v
		mz_y2 = cy + hh - mi_v
		gray[mz_y1:mz_y2, cx - hw:cx + hw] = 30
		score = omr_utils.bubble_reader.score_bubble_fast(
			gray, cx, cy, 10, geom)
		assert score > 0.1

	def test_out_of_bounds_negative(self) -> None:
		"""Out-of-bounds coordinates return -1."""
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		score = omr_utils.bubble_reader.score_bubble_fast(
			gray, -10, -10, 10)
		assert score == -1.0


#============================================
class TestRefineBubbleEdgesY:
	"""Tests for _refine_bubble_edges_y with synthetic bracket images."""

	def test_refines_shifted_bracket(self) -> None:
		"""Refinement snaps to bracket edges when template y is off.

		Creates a synthetic bracket at y=55 but passes template y=50.
		The refined y should be close to the actual bracket center.
		"""
		geom = _make_default_geom()
		gray = numpy.full((120, 120), 255, dtype=numpy.uint8)
		hw = geom["half_width"]
		hh = geom["half_height"]
		cx = 60
		# actual bracket center at y=55 (template thinks y=50)
		actual_cy = 55
		# paint dark bracket edges at actual position
		gray[actual_cy - hh:actual_cy - hh + 2, cx - hw:cx + hw] = 30
		gray[actual_cy + hh - 2:actual_cy + hh, cx - hw:cx + hw] = 30
		# call refinement with template estimate y=50
		refined_cy, top_y, bot_y = omr_utils.bubble_reader._refine_bubble_edges_y(
			gray, cx, 50, geom)
		# refined should be closer to actual center than template was
		template_error = abs(50 - actual_cy)
		refined_error = abs(refined_cy - actual_cy)
		assert refined_error < template_error

	def test_fallback_on_blank_area(self) -> None:
		"""Refinement returns template y when no edges are found.

		A uniform white image has no bracket edges, so the function
		should fall back to the original template position.
		"""
		geom = _make_default_geom()
		gray = numpy.full((120, 120), 255, dtype=numpy.uint8)
		refined_cy, top_y, bot_y = omr_utils.bubble_reader._refine_bubble_edges_y(
			gray, 60, 60, geom)
		assert refined_cy == 60

	def test_fallback_on_large_shift(self) -> None:
		"""Refinement rejects shifts exceeding refine_max_shift.

		Creates a bracket far from the template position; the
		function should keep the template position.
		"""
		geom = _make_default_geom()
		gray = numpy.full((200, 120), 255, dtype=numpy.uint8)
		hw = geom["half_width"]
		hh = geom["half_height"]
		cx = 60
		# actual bracket center at y=120, template at y=100
		# shift of 20 exceeds refine_max_shift (8)
		actual_cy = 120
		gray[actual_cy - hh:actual_cy - hh + 2, cx - hw:cx + hw] = 30
		gray[actual_cy + hh - 2:actual_cy + hh, cx - hw:cx + hw] = 30
		refined_cy, top_y, bot_y = omr_utils.bubble_reader._refine_bubble_edges_y(
			gray, cx, 100, geom)
		assert refined_cy == 100

	def test_rejects_wide_edge_separation(self) -> None:
		"""Edges with separation deviating >40% from expected are rejected.

		Simulates a column header producing edges much wider apart than
		a bubble. Should fall back to template position.
		"""
		geom = _make_default_geom()
		hh = geom["half_height"]
		hw = geom["half_width"]
		# create edges 4x expected separation apart (way more than 40%)
		gray = numpy.full((200, 120), 255, dtype=numpy.uint8)
		cx = 60
		cy = 100
		# place edges very far apart (simulating column header)
		wide_half = hh * 4
		gray[cy - wide_half:cy - wide_half + 2, cx - hw:cx + hw] = 30
		gray[cy + wide_half - 2:cy + wide_half, cx - hw:cx + hw] = 30
		refined_cy, top_y, bot_y = omr_utils.bubble_reader._refine_bubble_edges_y(
			gray, cx, cy, geom)
		# should fall back to template position
		assert refined_cy == cy
		assert top_y == cy - hh
		assert bot_y == cy + hh


#============================================
class TestValidateBubbleRect:
	"""Tests for _validate_bubble_rect area and aspect ratio checks."""

	def test_correct_rect_passes(self) -> None:
		"""A correctly-sized rectangle passes validation unchanged."""
		geom = _make_default_geom()
		hw = geom["half_width"]
		hh = geom["half_height"]
		cx, cy = 50, 50
		top_y = cy - hh
		bot_y = cy + hh
		left_x = cx - hw
		right_x = cx + hw
		result = omr_utils.bubble_reader._validate_bubble_rect(
			top_y, bot_y, left_x, right_x, cx, cy, geom)
		assert result == (top_y, bot_y, left_x, right_x, cx)

	def test_half_width_rejected(self) -> None:
		"""A rectangle half the expected width falls back to defaults.

		Simulates the phone photo failure where Sobel-x detects a
		letter stroke instead of the bracket arm, producing a
		rectangle covering only half the bubble.
		"""
		geom = _make_default_geom()
		hw = geom["half_width"]
		hh = geom["half_height"]
		cx, cy = 50, 50
		top_y = cy - hh
		bot_y = cy + hh
		# half-width rectangle (simulates letter edge detection)
		left_x = cx
		right_x = cx + hw
		result = omr_utils.bubble_reader._validate_bubble_rect(
			top_y, bot_y, left_x, right_x, cx, cy, geom)
		# should fall back to centered defaults
		assert result[2] == cx - hw  # default left
		assert result[3] == cx + hw  # default right
		assert result[4] == cx  # center unchanged

	def test_half_height_rejected(self) -> None:
		"""A rectangle half the expected height falls back to defaults."""
		geom = _make_default_geom()
		hw = geom["half_width"]
		hh = geom["half_height"]
		cx, cy = 50, 50
		left_x = cx - hw
		right_x = cx + hw
		# half-height rectangle
		top_y = cy
		bot_y = cy + hh
		result = omr_utils.bubble_reader._validate_bubble_rect(
			top_y, bot_y, left_x, right_x, cx, cy, geom)
		# should fall back to defaults
		assert result[0] == cy - hh  # default top
		assert result[1] == cy + hh  # default bottom


#============================================
class TestBracketEdgeMean:
	"""Tests for _compute_bracket_edge_mean on synthetic images."""

	def test_bracket_edge_mean_on_synthetic(self) -> None:
		"""Bracket edges dark and measurement zone white are measured.

		Creates a synthetic image with dark bracket edges (top/bottom
		of bubble box) and white center, then verifies that bracket
		edge mean is dark and edge mean is bright.
		"""
		geom = _make_default_geom()
		gray = numpy.full((100, 100), 255, dtype=numpy.uint8)
		hw = geom["half_width"]
		hh = geom["half_height"]
		beh = geom["bracket_edge_height"]
		cx, cy = 50, 50
		# compute edge positions
		top_y = cy - hh
		bot_y = cy + hh
		left_x = cx - hw
		right_x = cx + hw
		# paint dark bracket edges
		gray[top_y:top_y + beh, left_x:right_x] = 30
		gray[bot_y - beh:bot_y, left_x:right_x] = 30
		# bracket edge mean should be dark (low value)
		bracket_mean = omr_utils.bubble_reader._compute_bracket_edge_mean(
			gray, cx, cy, top_y, bot_y, left_x, right_x, geom)
		assert bracket_mean < 80
		# measurement zone excludes bracket edges via inset,
		# so edge_mean should be bright on this synthetic image
		edge_mean = omr_utils.bubble_reader._compute_edge_mean(
			gray, cx, cy, top_y, bot_y, left_x, right_x, geom)
		assert edge_mean > 200
		assert edge_mean > bracket_mean


#============================================
class TestFindAdaptiveThreshold:
	"""Tests for _find_adaptive_threshold."""

	def test_min_spread_floor_enforced(self) -> None:
		"""Threshold should never go below min_spread_floor.

		Creates spreads where the natural gap threshold would be very
		low, but min_spread_floor should clamp it up.
		"""
		# all spreads very low (unimodal), so gap-based threshold is tiny
		spreads = [(i, float(i) * 0.1) for i in range(1, 20)]
		threshold = omr_utils.bubble_reader._find_adaptive_threshold(
			spreads, min_spread_floor=15.0)
		assert threshold >= 15.0

	def test_bimodal_gap_detection(self) -> None:
		"""Clear bimodal distribution produces correct threshold.

		Blank questions cluster near zero spread, filled questions
		cluster near high spread. Threshold should be between them.
		"""
		# blank: spreads 1-5, filled: spreads 50-55
		blank = [(i, float(i)) for i in range(1, 6)]
		filled = [(i + 10, float(50 + i)) for i in range(1, 6)]
		spreads = blank + filled
		threshold = omr_utils.bubble_reader._find_adaptive_threshold(
			spreads, min_spread_floor=5.0)
		# threshold should be between the two populations
		assert threshold > 5.0
		assert threshold < 50.0

	def test_unimodal_uses_floor(self) -> None:
		"""Unimodal data with no significant gap uses floor."""
		# evenly spaced values: no single gap stands out
		spreads = [(i, float(i) * 2.0) for i in range(1, 50)]
		threshold = omr_utils.bubble_reader._find_adaptive_threshold(
			spreads, min_spread_floor=20.0)
		assert threshold >= 20.0


#============================================
class TestReadAnswers:
	"""Tests for read_answers with real scantron images."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	@pytest.fixture()
	def key_registered(self, template: dict) -> numpy.ndarray:
		"""Register and return the answer key image."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		return registered

	@pytest.fixture()
	def student_registered(self, template: dict) -> numpy.ndarray:
		"""Register and return the first discovered student image."""
		_skip_if_no_scantrons()
		students = _discover_student_images()
		if not students:
			pytest.skip("no student scantron images found")
		img = omr_utils.image_registration.load_image(students[0])
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		return registered

	def test_returns_100_results(self, template: dict,
		student_registered: numpy.ndarray) -> None:
		"""read_answers returns exactly 100 question results."""
		results = omr_utils.bubble_reader.read_answers(
			student_registered, template)
		assert len(results) == 100

	def test_result_keys(self, template: dict,
		student_registered: numpy.ndarray) -> None:
		"""Each result has required keys including refined positions and edges."""
		results = omr_utils.bubble_reader.read_answers(
			student_registered, template)
		first = results[0]
		assert "question" in first
		assert "answer" in first
		assert "scores" in first
		assert "flags" in first
		assert "positions" in first
		assert "edges" in first
		# positions should have entries for each choice
		for choice in ["A", "B", "C", "D", "E"]:
			assert choice in first["positions"]
			px, py = first["positions"][choice]
			assert isinstance(px, int)
			assert isinstance(py, int)
		# edges should have (top_y, bot_y, left_x, right_x) per choice
		for choice in ["A", "B", "C", "D", "E"]:
			assert choice in first["edges"]
			edge_tuple = first["edges"][choice]
			assert len(edge_tuple) == 4

	def test_q28_is_e(self, template: dict,
		key_registered: numpy.ndarray) -> None:
		"""Question 28 answer should be E (known from all forms)."""
		results = omr_utils.bubble_reader.read_answers(
			key_registered, template)
		# find Q28
		q28 = [r for r in results if r["question"] == 28][0]
		assert q28["answer"] == "E"

	def test_most_questions_answered(self, template: dict,
		student_registered: numpy.ndarray) -> None:
		"""At least 50 questions should have answers (non-blank)."""
		results = omr_utils.bubble_reader.read_answers(
			student_registered, template)
		answered = sum(1 for r in results if r["answer"])
		assert answered >= 50

	def test_valid_choices_only(self, template: dict,
		student_registered: numpy.ndarray) -> None:
		"""All detected answers should be valid choices (A-E) or blank."""
		results = omr_utils.bubble_reader.read_answers(
			student_registered, template)
		valid = {"A", "B", "C", "D", "E", ""}
		for r in results:
			assert r["answer"] in valid

	def test_key_detects_at_least_71(self, template: dict,
		key_registered: numpy.ndarray) -> None:
		"""Answer key should detect at least 71 non-blank answers.

		Regression test: previously only 66 were detected because
		Q1, Q2, Q51, Q52, Q53 were missed due to column-header
		edge confusion with 6px-tall measurement zones.
		"""
		results = omr_utils.bubble_reader.read_answers(
			key_registered, template)
		answered = sum(1 for r in results if r["answer"])
		assert answered >= 71

	def test_key_q1_detected(self, template: dict,
		key_registered: numpy.ndarray) -> None:
		"""Q1 should be detected as non-blank on the answer key.

		Regression test: Q1 was previously missed because Sobel-y
		found column header edges instead of bubble bracket edges.
		"""
		results = omr_utils.bubble_reader.read_answers(
			key_registered, template)
		q1 = [r for r in results if r["question"] == 1][0]
		assert q1["answer"] != "", "Q1 should not be blank"

	def test_all_students_detect_at_least_71(self, template: dict) -> None:
		"""All student images should detect at least 71 non-blank answers.

		Regression test: phone photo previously detected only 69/71
		due to measurement zone landing in white gap between rows
		for Q44 and Q49. Row linearity and brightness checks fix this.
		"""
		_skip_if_no_scantrons()
		students = _discover_student_images()
		if not students:
			pytest.skip("no student scantron images found")
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		for image_path in students:
			img = omr_utils.image_registration.load_image(image_path)
			registered = omr_utils.image_registration.register_image(
				img, canon_w, canon_h)
			results = omr_utils.bubble_reader.read_answers(
				registered, template)
			answered = sum(1 for r in results if r["answer"])
			basename = os.path.basename(image_path)
			assert answered >= 71, (
				f"{basename}: detected {answered}/71, expected >= 71")


#============================================
class TestRowLinearity:
	"""Tests for _check_row_linearity median-based outlier detection."""

	def test_aligned_row_no_outliers(self) -> None:
		"""A row with consistent y-centers has no outliers."""
		choices = ["A", "B", "C", "D", "E"]
		q_choices = {
			"A": {"px": 100, "refined_cy": 500},
			"B": {"px": 150, "refined_cy": 501},
			"C": {"px": 200, "refined_cy": 500},
			"D": {"px": 250, "refined_cy": 499},
			"E": {"px": 300, "refined_cy": 500},
		}
		outliers = omr_utils.bubble_reader._check_row_linearity(
			q_choices, choices)
		assert outliers == []

	def test_single_outlier_detected(self) -> None:
		"""A single choice with y-center far from the row is flagged."""
		choices = ["A", "B", "C", "D", "E"]
		q_choices = {
			"A": {"px": 100, "refined_cy": 500},
			"B": {"px": 150, "refined_cy": 488},
			"C": {"px": 200, "refined_cy": 500},
			"D": {"px": 250, "refined_cy": 500},
			"E": {"px": 300, "refined_cy": 500},
		}
		outliers = omr_utils.bubble_reader._check_row_linearity(
			q_choices, choices)
		# B should be flagged as outlier
		outlier_choices = [c for c, _ in outliers]
		assert "B" in outlier_choices
		assert len(outliers) == 1

	def test_two_outliers_detected(self) -> None:
		"""Two minority bad detections are caught while majority is kept."""
		choices = ["A", "B", "C", "D", "E"]
		# simulates the phone photo Q45 pattern: B and D 12px off
		q_choices = {
			"A": {"px": 100, "refined_cy": 1838},
			"B": {"px": 150, "refined_cy": 1826},
			"C": {"px": 200, "refined_cy": 1839},
			"D": {"px": 250, "refined_cy": 1826},
			"E": {"px": 300, "refined_cy": 1838},
		}
		outliers = omr_utils.bubble_reader._check_row_linearity(
			q_choices, choices)
		outlier_choices = [c for c, _ in outliers]
		assert "B" in outlier_choices
		assert "D" in outlier_choices
		# A, C, E should NOT be flagged
		assert "A" not in outlier_choices
		assert "C" not in outlier_choices
		assert "E" not in outlier_choices


#============================================
class TestRefinementShiftData:
	"""Tests for refinement-shift diagnostics in debug overlay."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template for shift diagnostics tests."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_local_similarity_flags_outlier_shift(self, template: dict) -> None:
		"""A local shift outlier should be marked as not locally similar."""
		choices = template["answers"]["choices"]
		results = []
		for q_num in range(1, 6):
			positions = {}
			scores = {}
			for choice in choices:
				norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
					template, q_num, choice)
				px, py = omr_utils.template_loader.to_pixels(
					norm_x, norm_y, 1700, 2200)
				# mostly small right shift, with one outlier on q3/A
				if q_num == 3 and choice == "A":
					positions[choice] = (px + 8, py)
				else:
					positions[choice] = (px + 1, py)
				scores[choice] = 0.0
			entry = {
				"question": q_num,
				"answer": "",
				"scores": scores,
				"flags": "BLANK",
				"positions": positions,
				"edges": {},
			}
			results.append(entry)
		shift_data = omr_utils.debug_drawing._compute_refinement_shift_data(
			results, template, 1700, 2200)
		assert shift_data[(3, "A")]["local_ok"] is False
		assert shift_data[(2, "A")]["local_ok"] is True


#============================================
class TestRowBrightness:
	"""Tests for _check_row_brightness all-white detection."""

	def test_normal_row_passes(self) -> None:
		"""A row with mixed brightness values is not flagged."""
		choices = ["A", "B", "C", "D", "E"]
		edge_means = {"A": 230.0, "B": 120.0, "C": 225.0, "D": 200.0, "E": 215.0}
		result = omr_utils.bubble_reader._check_row_brightness(
			edge_means, choices)
		assert result is False

	def test_all_white_flagged(self) -> None:
		"""A row where all choices are white (>220) is flagged."""
		choices = ["A", "B", "C", "D", "E"]
		edge_means = {"A": 235.0, "B": 240.0, "C": 230.0, "D": 225.0, "E": 238.0}
		result = omr_utils.bubble_reader._check_row_brightness(
			edge_means, choices)
		assert result is True
