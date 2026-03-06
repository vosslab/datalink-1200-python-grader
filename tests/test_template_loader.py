"""Unit tests for omr_utils.template_loader."""

# Standard Library
import os

# PIP3 modules
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")


#============================================
class TestLoadTemplate:
	"""Tests for load_template function."""

	def test_loads_valid_template(self) -> None:
		"""Template file loads without errors."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		assert isinstance(template, dict)

	def test_required_keys_present(self) -> None:
		"""Template contains all required top-level keys."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		assert "form" in template
		assert "canonical" in template
		assert "student_id" in template
		assert "answers" in template

	def test_missing_file_raises(self) -> None:
		"""Missing template file raises FileNotFoundError."""
		with pytest.raises(FileNotFoundError):
			omr_utils.template_loader.load_template("/nonexistent/path.yaml")

	def test_form_metadata(self) -> None:
		"""Template form section has expected metadata."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		assert template["form"]["orientation"] == "portrait"
		assert "DataLink" in template["form"]["name"]

	def test_answer_columns(self) -> None:
		"""Template has left and right answer columns."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		answers = template["answers"]
		assert answers["num_questions"] == 100
		assert answers["left_column"]["question_range"] == [1, 50]
		assert answers["right_column"]["question_range"] == [51, 100]
		assert template["template_version"] == 3
		assert "bubble_shape" in answers
		assert "bubble_geometry" not in answers

	def test_student_id_config(self) -> None:
		"""Template student_id section is correctly structured."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		sid = template["student_id"]
		assert sid["num_digits"] == 9
		assert sid["grid"]["digit_values"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


#============================================
class TestTemplateMigration:
	"""Tests for v1-to-v2 template migration helpers."""

	def test_migrate_v1_geometry_to_v2_shape(self) -> None:
		"""Legacy bubble_geometry fields migrate to bubble_shape contract."""
		template_v1 = {
			"form": {"name": "x", "orientation": "portrait"},
			"canonical": {"width_px": 1700, "height_px": 2200},
			"student_id": {
				"num_digits": 9,
				"grid": {
					"first_digit_x": 0.1,
					"digit_spacing_x": 0.05,
					"digit_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
					"first_value_y": 0.1,
					"value_spacing_y": 0.01,
				},
				"bubble_radius": 0.006,
			},
			"answers": {
				"num_questions": 100,
				"choices": ["A", "B", "C", "D", "E"],
				"bubble_radius": 0.006,
				"bubble_geometry": {
					"half_width": 0.01765,
					"half_height": 0.0025,
				},
				"left_column": {
					"question_range": [1, 50],
					"first_question_y": 0.2,
					"question_spacing_y": 0.014,
					"choice_x": {"A": 0.1, "B": 0.15, "C": 0.2, "D": 0.25, "E": 0.3},
				},
				"right_column": {
					"question_range": [51, 100],
					"first_question_y": 0.2,
					"question_spacing_y": 0.014,
					"choice_x": {"A": 0.4, "B": 0.45, "C": 0.5, "D": 0.55, "E": 0.6},
				},
			},
		}
		migrated = omr_utils.template_loader.migrate_template_to_v2(template_v1)
		assert migrated["template_version"] == 2
		assert "bubble_shape" in migrated["answers"]
		assert "bubble_geometry" not in migrated["answers"]
		shape = migrated["answers"]["bubble_shape"]
		assert shape["aspect_ratio"] > 5.0
		assert shape["target_area_px_at_canonical"] > 500.0

	def test_geometry_derived_from_shape_matches_canonical(self) -> None:
		"""Derived canonical geometry remains close to known working values."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		geom = omr_utils.template_loader.get_bubble_geometry_px(
			template, 1700, 2200)
		assert 29.0 <= geom["half_width"] <= 31.0
		assert 5.0 <= geom["half_height"] <= 6.0
		assert 10.0 <= geom["center_exclusion"] <= 12.5


#============================================
class TestGetBubbleCoords:
	"""Tests for get_bubble_coords function."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template once for all tests in this class."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_question_1_a(self, template: dict) -> None:
		"""Question 1 choice A returns valid normalized coords."""
		norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(template, 1, "A")
		# should be in the left column, near top of answer area
		assert 0.0 < norm_x < 0.5
		assert 0.2 < norm_y < 0.4

	def test_question_50_e(self, template: dict) -> None:
		"""Question 50 choice E returns valid coords in left column."""
		norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(template, 50, "E")
		assert 0.0 < norm_x < 0.5
		assert norm_y > 0.5

	def test_question_51_a(self, template: dict) -> None:
		"""Question 51 choice A is in right column."""
		norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(template, 51, "A")
		assert norm_x > 0.4
		assert 0.2 < norm_y < 0.4

	def test_question_100_e(self, template: dict) -> None:
		"""Question 100 choice E is at bottom of right column."""
		norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(template, 100, "E")
		assert norm_x > 0.4
		assert norm_y > 0.5

	def test_all_100_questions(self, template: dict) -> None:
		"""All 100 questions and 5 choices return coords without error."""
		choices = ["A", "B", "C", "D", "E"]
		for q in range(1, 101):
			for c in choices:
				nx, ny = omr_utils.template_loader.get_bubble_coords(template, q, c)
				assert 0.0 <= nx <= 1.0
				assert 0.0 <= ny <= 1.0

	def test_invalid_question_raises(self, template: dict) -> None:
		"""Question number out of range raises ValueError."""
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_bubble_coords(template, 0, "A")
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_bubble_coords(template, 101, "A")

	def test_invalid_choice_raises(self, template: dict) -> None:
		"""Invalid choice letter raises ValueError."""
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_bubble_coords(template, 1, "F")

	def test_choices_increase_in_x(self, template: dict) -> None:
		"""Choices A through E have increasing x coordinates."""
		choices = ["A", "B", "C", "D", "E"]
		for q in [1, 25, 50, 51, 75, 100]:
			x_vals = []
			for c in choices:
				nx, _ = omr_utils.template_loader.get_bubble_coords(template, q, c)
				x_vals.append(nx)
			for i in range(len(x_vals) - 1):
				assert x_vals[i] < x_vals[i + 1]

	def test_questions_increase_in_y(self, template: dict) -> None:
		"""Questions within a column have increasing y coordinates."""
		# left column
		for q in range(1, 50):
			_, y1 = omr_utils.template_loader.get_bubble_coords(template, q, "A")
			_, y2 = omr_utils.template_loader.get_bubble_coords(template, q + 1, "A")
			assert y1 < y2
		# right column
		for q in range(51, 100):
			_, y1 = omr_utils.template_loader.get_bubble_coords(template, q, "A")
			_, y2 = omr_utils.template_loader.get_bubble_coords(template, q + 1, "A")
			assert y1 < y2


#============================================
class TestGetStudentIdCoords:
	"""Tests for get_student_id_coords function."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template once for all tests in this class."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_first_digit_zero(self, template: dict) -> None:
		"""First digit position, value 0, returns valid coords."""
		nx, ny = omr_utils.template_loader.get_student_id_coords(template, 0, 0)
		assert 0.0 < nx < 0.5
		assert 0.0 < ny < 0.3

	def test_all_90_bubbles(self, template: dict) -> None:
		"""All 9 digits x 10 values return valid coords."""
		for d in range(9):
			for v in range(10):
				nx, ny = omr_utils.template_loader.get_student_id_coords(template, d, v)
				assert 0.0 <= nx <= 1.0
				assert 0.0 <= ny <= 1.0

	def test_digits_increase_in_x(self, template: dict) -> None:
		"""Digit positions increase in x coordinate left to right."""
		for v in [0, 5, 9]:
			prev_x = -1.0
			for d in range(9):
				nx, _ = omr_utils.template_loader.get_student_id_coords(template, d, v)
				assert nx > prev_x
				prev_x = nx

	def test_values_increase_in_y(self, template: dict) -> None:
		"""Digit values 0-9 increase in y coordinate top to bottom."""
		for d in [0, 4, 8]:
			prev_y = -1.0
			for v in range(10):
				_, ny = omr_utils.template_loader.get_student_id_coords(template, d, v)
				assert ny > prev_y
				prev_y = ny

	def test_invalid_digit_raises(self, template: dict) -> None:
		"""Invalid digit position raises ValueError."""
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_student_id_coords(template, -1, 0)
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_student_id_coords(template, 9, 0)

	def test_invalid_value_raises(self, template: dict) -> None:
		"""Invalid digit value raises ValueError."""
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_student_id_coords(template, 0, -1)
		with pytest.raises(ValueError):
			omr_utils.template_loader.get_student_id_coords(template, 0, 10)


#============================================
class TestToPixels:
	"""Tests for to_pixels function."""

	def test_center(self) -> None:
		"""Center of normalized space maps to center pixel."""
		px, py = omr_utils.template_loader.to_pixels(0.5, 0.5, 1000, 2000)
		assert px == 500
		assert py == 1000

	def test_origin(self) -> None:
		"""Origin maps to pixel (0, 0)."""
		px, py = omr_utils.template_loader.to_pixels(0.0, 0.0, 1000, 2000)
		assert px == 0
		assert py == 0

	def test_full_extent(self) -> None:
		"""(1.0, 1.0) maps to image dimensions."""
		px, py = omr_utils.template_loader.to_pixels(1.0, 1.0, 1700, 2200)
		assert px == 1700
		assert py == 2200


#============================================
class TestGetBubbleRadiusPx:
	"""Tests for get_bubble_radius_px function."""

	def test_returns_positive(self) -> None:
		"""Bubble radius is a positive integer."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		radius = omr_utils.template_loader.get_bubble_radius_px(template, 1700, 2200)
		assert radius > 0

	def test_minimum_radius(self) -> None:
		"""Bubble radius is at least 3 even for tiny images."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		radius = omr_utils.template_loader.get_bubble_radius_px(template, 10, 10)
		assert radius >= 3


#============================================
class TestGetAllCoords:
	"""Tests for get_all_question_coords and get_all_student_id_coords."""

	@pytest.fixture()
	def template(self) -> dict:
		"""Load template once for all tests in this class."""
		loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		return loaded

	def test_all_question_coords_count(self, template: dict) -> None:
		"""get_all_question_coords returns 500 entries (100 questions x 5 choices)."""
		coords = omr_utils.template_loader.get_all_question_coords(template)
		assert len(coords) == 500

	def test_all_question_coords_keys(self, template: dict) -> None:
		"""Each entry has required keys."""
		coords = omr_utils.template_loader.get_all_question_coords(template)
		first = coords[0]
		assert "question" in first
		assert "choice" in first
		assert "norm_x" in first
		assert "norm_y" in first

	def test_all_student_id_coords_count(self, template: dict) -> None:
		"""get_all_student_id_coords returns 90 entries (9 digits x 10 values)."""
		coords = omr_utils.template_loader.get_all_student_id_coords(template)
		assert len(coords) == 90

	def test_all_student_id_coords_keys(self, template: dict) -> None:
		"""Each entry has required keys."""
		coords = omr_utils.template_loader.get_all_student_id_coords(template)
		first = coords[0]
		assert "digit" in first
		assert "value" in first
		assert "norm_x" in first
		assert "norm_y" in first


#============================================
class TestV3Migration:
	"""Tests for v3 template migration with timing mark indices."""

	def test_v3_yaml_loads_with_coordinates(self) -> None:
		"""V3 YAML (mark indices only) gets coordinate fields at load time."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		left = template["answers"]["left_column"]
		# v3 mark index fields present
		assert "first_row_mark_index" in left
		assert "row_spacing_marks" in left
		assert "choice_mark_indices" in left
		# v2 coordinate fields computed from mark indices
		assert "first_question_y" in left
		assert "question_spacing_y" in left
		assert "choice_x" in left

	def test_v3_student_id_has_both_formats(self) -> None:
		"""V3 student ID grid has both mark indices and coordinates."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		grid = template["student_id"]["grid"]
		# v3 mark index fields
		assert "first_digit_mark_index" in grid
		assert "digit_spacing_marks" in grid
		assert "first_value_mark_index" in grid
		assert "value_spacing_marks" in grid
		# v2 coordinate fields
		assert "first_digit_x" in grid
		assert "digit_spacing_x" in grid
		assert "first_value_y" in grid
		assert "value_spacing_y" in grid

	def test_v2_to_v3_roundtrip_answer_coords(self) -> None:
		"""V2 coordinates survive v3 migration with sub-pixel accuracy."""
		# original v2 values
		v2_left_first_y = 0.2164
		v2_left_spacing = 0.01400
		v2_left_choice_a = 0.1212
		# load v3 template (which was migrated from v2 values)
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		left = template["answers"]["left_column"]
		# computed coordinates should match v2 originals closely
		assert abs(left["first_question_y"] - v2_left_first_y) < 0.0001
		assert abs(left["question_spacing_y"] - v2_left_spacing) < 0.0001
		assert abs(left["choice_x"]["A"] - v2_left_choice_a) < 0.0001

	def test_v2_to_v3_roundtrip_student_id(self) -> None:
		"""V2 student ID coordinates survive v3 migration."""
		v2_first_digit_x = 0.1218
		v2_digit_spacing = 0.0499
		v2_first_value_y = 0.0745
		v2_value_spacing = 0.0128
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		grid = template["student_id"]["grid"]
		assert abs(grid["first_digit_x"] - v2_first_digit_x) < 0.0001
		assert abs(grid["digit_spacing_x"] - v2_digit_spacing) < 0.0001
		assert abs(grid["first_value_y"] - v2_first_value_y) < 0.0001
		assert abs(grid["value_spacing_y"] - v2_value_spacing) < 0.0001

	def test_v2_yaml_backward_compat(self) -> None:
		"""A v2-style template dict migrates to v3 with mark indices."""
		template_v2 = {
			"form": {"name": "test", "orientation": "portrait"},
			"canonical": {"width_px": 1700, "height_px": 2200},
			"template_version": 2,
			"timing_marks": {
				"top_edge": {
					"y": 0.012, "start_x": 0.04, "end_x": 0.96,
					"expected_count": 53,
				},
				"left_edge": {
					"x": 0.018, "start_y": 0.067, "end_y": 0.91,
					"expected_count": 60,
				},
			},
			"student_id": {
				"num_digits": 9,
				"grid": {
					"first_digit_x": 0.1218,
					"digit_spacing_x": 0.0499,
					"digit_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
					"first_value_y": 0.0745,
					"value_spacing_y": 0.0128,
				},
				"bubble_radius": 0.006,
			},
			"answers": {
				"num_questions": 100,
				"choices": ["A", "B", "C", "D", "E"],
				"bubble_radius": 0.006,
				"bubble_shape": {
					"aspect_ratio": 5.454545,
					"target_area_px_at_canonical": 660.0,
				},
				"left_column": {
					"question_range": [1, 50],
					"first_question_y": 0.2164,
					"question_spacing_y": 0.014,
					"choice_x": {
						"A": 0.12, "B": 0.17, "C": 0.22,
						"D": 0.27, "E": 0.32,
					},
				},
				"right_column": {
					"question_range": [51, 100],
					"first_question_y": 0.2164,
					"question_spacing_y": 0.014,
					"choice_x": {
						"A": 0.47, "B": 0.52, "C": 0.57,
						"D": 0.62, "E": 0.67,
					},
				},
			},
		}
		migrated = omr_utils.template_loader.migrate_template_to_v3(template_v2)
		assert migrated["template_version"] == 3
		left = migrated["answers"]["left_column"]
		# mark indices should be computed
		assert "first_row_mark_index" in left
		assert "row_spacing_marks" in left
		assert "choice_mark_indices" in left
		# original coordinates preserved
		assert left["first_question_y"] == 0.2164
		assert left["question_spacing_y"] == 0.014
