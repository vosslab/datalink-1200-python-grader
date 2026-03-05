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

	def test_student_id_config(self) -> None:
		"""Template student_id section is correctly structured."""
		template = omr_utils.template_loader.load_template(TEMPLATE_PATH)
		sid = template["student_id"]
		assert sid["num_digits"] == 9
		assert sid["grid"]["digit_values"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
