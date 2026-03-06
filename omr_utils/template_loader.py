"""Load and query DataLink 1200 form geometry from a YAML template."""

# Standard Library
import os

# PIP3 modules
import yaml


#============================================
def load_template(yaml_path: str) -> dict:
	"""Load and validate a form geometry YAML template.

	Args:
		yaml_path: path to the YAML template file

	Returns:
		parsed template dictionary

	Raises:
		FileNotFoundError: if the YAML file does not exist
		ValueError: if required keys are missing
	"""
	if not os.path.isfile(yaml_path):
		raise FileNotFoundError(f"template not found: {yaml_path}")
	with open(yaml_path, "r") as fh:
		template = yaml.safe_load(fh)
	# validate required top-level sections
	required_keys = ("form", "canonical", "student_id", "answers")
	for key in required_keys:
		if key not in template:
			raise ValueError(f"template missing required key: {key}")
	# validate answer section
	answers = template["answers"]
	if "left_column" not in answers or "right_column" not in answers:
		raise ValueError("template answers must have left_column and right_column")
	return template


#============================================
def get_bubble_coords(template: dict, question: int, choice: str) -> tuple:
	"""Return normalized (x, y) center for a given question and choice bubble.

	Args:
		template: loaded template dictionary
		question: question number (1 to num_questions)
		choice: answer choice letter (A through E)

	Returns:
		tuple of (norm_x, norm_y) in range 0.0 to 1.0

	Raises:
		ValueError: if question or choice is out of range
	"""
	answers = template["answers"]
	num_q = answers["num_questions"]
	choices = answers["choices"]
	if question < 1 or question > num_q:
		raise ValueError(f"question {question} out of range 1-{num_q}")
	if choice not in choices:
		raise ValueError(f"choice '{choice}' not in {choices}")
	# determine which column this question belongs to
	left = answers["left_column"]
	right = answers["right_column"]
	left_range = left["question_range"]
	right_range = right["question_range"]
	if left_range[0] <= question <= left_range[1]:
		col = left
		# offset within column (0-indexed)
		offset = question - left_range[0]
	elif right_range[0] <= question <= right_range[1]:
		col = right
		offset = question - right_range[0]
	else:
		raise ValueError(f"question {question} not in any column range")
	# compute y from first_question_y + offset * spacing
	norm_y = col["first_question_y"] + offset * col["question_spacing_y"]
	# compute x from choice position
	norm_x = col["choice_x"][choice]
	return (norm_x, norm_y)


#============================================
def get_student_id_coords(template: dict, digit: int, value: int) -> tuple:
	"""Return normalized (x, y) center for a student ID digit bubble.

	Args:
		template: loaded template dictionary
		digit: digit position (0 to num_digits-1, left to right)
		value: digit value (0-9)

	Returns:
		tuple of (norm_x, norm_y) in range 0.0 to 1.0

	Raises:
		ValueError: if digit or value is out of range
	"""
	sid = template["student_id"]
	num_digits = sid["num_digits"]
	if digit < 0 or digit >= num_digits:
		raise ValueError(f"digit {digit} out of range 0-{num_digits - 1}")
	if value < 0 or value > 9:
		raise ValueError(f"value {value} out of range 0-9")
	grid = sid["grid"]
	# x position: first_digit_x + digit * digit_spacing_x
	norm_x = grid["first_digit_x"] + digit * grid["digit_spacing_x"]
	# y position: first_value_y + value * value_spacing_y
	norm_y = grid["first_value_y"] + value * grid["value_spacing_y"]
	return (norm_x, norm_y)


#============================================
def to_pixels(norm_x: float, norm_y: float, width: int, height: int) -> tuple:
	"""Convert normalized coordinates to pixel coordinates.

	Args:
		norm_x: normalized x (0.0 to 1.0)
		norm_y: normalized y (0.0 to 1.0)
		width: image width in pixels
		height: image height in pixels

	Returns:
		tuple of (pixel_x, pixel_y) as integers
	"""
	px = int(round(norm_x * width))
	py = int(round(norm_y * height))
	return (px, py)


#============================================
def get_bubble_radius_px(template: dict, width: int, height: int) -> int:
	"""Return bubble radius in pixels for the answer bubbles.

	Args:
		template: loaded template dictionary
		width: image width in pixels
		height: image height in pixels

	Returns:
		radius in pixels (based on smaller dimension for safety)
	"""
	norm_radius = template["answers"]["bubble_radius"]
	# use the smaller dimension to compute pixel radius
	min_dim = min(width, height)
	radius_px = int(round(norm_radius * min_dim))
	return max(radius_px, 3)


#============================================
def get_all_question_coords(template: dict) -> list:
	"""Return all bubble positions for the answer grid.

	Returns:
		list of dicts with keys: question, choice, norm_x, norm_y
		sorted by question number then choice letter
	"""
	answers = template["answers"]
	num_q = answers["num_questions"]
	choices = answers["choices"]
	coords = []
	for q_num in range(1, num_q + 1):
		for choice in choices:
			norm_x, norm_y = get_bubble_coords(template, q_num, choice)
			entry = {
				"question": q_num,
				"choice": choice,
				"norm_x": norm_x,
				"norm_y": norm_y,
			}
			coords.append(entry)
	return coords


#============================================
def get_bubble_geometry_px(template: dict, width: int, height: int) -> dict:
	"""Convert normalized bubble geometry to pixel values for a given image size.

	Reads the bubble_geometry section from the template and scales each
	value to pixel coordinates. Falls back to hardcoded defaults if the
	section is absent from the template.

	Args:
		template: loaded template dictionary
		width: image width in pixels
		height: image height in pixels

	Returns:
		dict with pixel values for all bubble geometry parameters
	"""
	# default normalized values (match YAML template at 1700x2200)
	defaults = {
		"half_width": 0.01765,
		"half_height": 0.00250,
		"center_exclusion": 0.00647,
		"bracket_edge_height": 0.00091,
		"measurement_inset_v": 0.00091,
		"measurement_inset_h": 0.00176,
		"refine_max_shift": 0.00364,
		"refine_pad_v": 0.00364,
		"refine_pad_h": 0.00471,
	}
	# read from template if present, else use defaults
	bg = template.get("answers", {}).get("bubble_geometry", defaults)
	# horizontal values scale with width, vertical with height
	# return float values; consumers use int() only at array-slicing boundaries
	geom = {
		"half_width": max(1.0, round(bg.get("half_width", defaults["half_width"]) * width, 1)),
		"half_height": max(1.0, round(bg.get("half_height", defaults["half_height"]) * height, 1)),
		"center_exclusion": max(1.0, round(bg.get("center_exclusion", defaults["center_exclusion"]) * width, 1)),
		"bracket_edge_height": max(1.0, round(bg.get("bracket_edge_height", defaults["bracket_edge_height"]) * height, 1)),
		"measurement_inset_v": max(1.0, round(bg.get("measurement_inset_v", defaults["measurement_inset_v"]) * height, 1)),
		"measurement_inset_h": max(1.0, round(bg.get("measurement_inset_h", defaults["measurement_inset_h"]) * width, 1)),
		"refine_max_shift": max(1.0, round(bg.get("refine_max_shift", defaults["refine_max_shift"]) * height, 1)),
		"refine_pad_v": max(1.0, round(bg.get("refine_pad_v", defaults["refine_pad_v"]) * height, 1)),
		"refine_pad_h": max(1.0, round(bg.get("refine_pad_h", defaults["refine_pad_h"]) * width, 1)),
	}
	return geom


#============================================
def get_all_student_id_coords(template: dict) -> list:
	"""Return all bubble positions for the student ID grid.

	Returns:
		list of dicts with keys: digit, value, norm_x, norm_y
		sorted by digit position then value
	"""
	sid = template["student_id"]
	num_digits = sid["num_digits"]
	coords = []
	for d in range(num_digits):
		for v in range(10):
			norm_x, norm_y = get_student_id_coords(template, d, v)
			entry = {
				"digit": d,
				"value": v,
				"norm_x": norm_x,
				"norm_y": norm_y,
			}
			coords.append(entry)
	return coords
