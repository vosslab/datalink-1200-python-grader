"""Load and query DataLink 1200 form geometry from a YAML template."""

# Standard Library
import os
import copy

# PIP3 modules
import yaml

# local repo modules
import omr_utils.timing_mark_anchors


#============================================
def _validate_required_sections(template: dict) -> None:
	"""Validate required top-level sections exist."""
	required_keys = ("form", "student_id", "answers")
	for key in required_keys:
		if key not in template:
			raise ValueError(f"template missing required key: {key}")


#============================================
def _validate_answer_columns(answers: dict) -> None:
	"""Validate required answer column sections exist."""
	if "left_column" not in answers or "right_column" not in answers:
		raise ValueError("template answers must have left_column and right_column")


#============================================
def _compute_shape_from_v1_geometry(template: dict) -> dict:
	"""Compute v2 shape fields from v1 bubble_geometry settings."""
	answers = template["answers"]
	# extract aspect ratio from v1 bubble_geometry if present
	bubble_geom = answers.get("bubble_geometry", {})
	half_w = float(bubble_geom.get("half_width", 30.0))
	half_h = float(bubble_geom.get("half_height", 5.5))
	aspect_ratio = half_w / half_h if half_h > 0 else 5.45
	shape = {
		"aspect_ratio": round(aspect_ratio, 6),
	}
	return shape


#============================================
def migrate_template_to_v2(template: dict) -> dict:
	"""Return a v2 template dict from either v1 or v2 input data.

	V2 keeps the same anchor/layout structure but replaces large
	per-geometry parameter surfaces with a minimal shape contract:
	- aspect_ratio

	Args:
		template: raw template dictionary loaded from YAML

	Returns:
		migrated template dictionary with template_version=2
	"""
	migrated = copy.deepcopy(template)
	_validate_required_sections(migrated)
	answers = migrated["answers"]
	_validate_answer_columns(answers)
	version = int(migrated.get("template_version", 1))
	if version >= 2 and "bubble_shape" in answers:
		migrated["template_version"] = 2
		return migrated
	shape = answers.get("bubble_shape")
	if shape is None:
		shape = _compute_shape_from_v1_geometry(migrated)
	answers["bubble_shape"] = shape
	if "bubble_geometry" in answers:
		del answers["bubble_geometry"]
	migrated["template_version"] = 2
	return migrated


#============================================
def _ensure_mark_indices(template: dict) -> None:
	"""Add v3 mark index fields to columns and student ID if missing.

	Computes fractional timing mark indices from v2 normalized
	coordinate fields using the timing mark edge definitions.

	Args:
		template: template dict with timing_marks and v2 coordinate fields
	"""
	timing = template.get("timing_marks", {})
	left_edge = timing.get("left_edge", {})
	top_edge = timing.get("top_edge", {})
	answers = template["answers"]
	# compute mark step sizes for spacing conversion
	left_step = ((left_edge["end_y"] - left_edge["start_y"])
		/ max(1, left_edge["expected_count"] - 1))
	top_step = ((top_edge["end_x"] - top_edge["start_x"])
		/ max(1, top_edge["expected_count"] - 1))
	# answer columns
	for col_key in ("left_column", "right_column"):
		col = answers[col_key]
		if "first_row_mark_index" in col:
			continue
		# convert y-coordinate to left mark index
		first_row_idx = omr_utils.timing_mark_anchors.normalized_to_mark_index(
			col["first_question_y"],
			left_edge["start_y"], left_edge["end_y"],
			left_edge["expected_count"])
		# convert y-spacing to mark spacing
		row_spacing = col["question_spacing_y"] / left_step
		# convert each choice x-coordinate to top mark index
		choice_indices = {}
		choice_x = col.get("choice_x", {})
		if not choice_x and "choice_columns" in col:
			# derive choice_x from integer column indices
			for letter, col_idx in col["choice_columns"].items():
				norm_x = omr_utils.timing_mark_anchors.mark_index_to_normalized(
					float(col_idx),
					top_edge["start_x"], top_edge["end_x"],
					top_edge["expected_count"])
				choice_x[letter] = round(norm_x, 6)
			col["choice_x"] = choice_x
		for letter, norm_x in choice_x.items():
			idx = omr_utils.timing_mark_anchors.normalized_to_mark_index(
				norm_x,
				top_edge["start_x"], top_edge["end_x"],
				top_edge["expected_count"])
			choice_indices[letter] = round(idx, 4)
		col["first_row_mark_index"] = round(first_row_idx, 4)
		col["row_spacing_marks"] = round(row_spacing, 4)
		col["choice_mark_indices"] = choice_indices
	# student ID grid
	grid = template["student_id"]["grid"]
	if "first_digit_mark_index" not in grid:
		# digit x-positions use top marks
		first_digit_idx = omr_utils.timing_mark_anchors.normalized_to_mark_index(
			grid["first_digit_x"],
			top_edge["start_x"], top_edge["end_x"],
			top_edge["expected_count"])
		digit_spacing = grid["digit_spacing_x"] / top_step
		# digit values y-positions use left marks
		first_value_idx = omr_utils.timing_mark_anchors.normalized_to_mark_index(
			grid["first_value_y"],
			left_edge["start_y"], left_edge["end_y"],
			left_edge["expected_count"])
		value_spacing = grid["value_spacing_y"] / left_step
		grid["first_digit_mark_index"] = round(first_digit_idx, 4)
		grid["digit_spacing_marks"] = round(digit_spacing, 4)
		grid["first_value_mark_index"] = round(first_value_idx, 4)
		grid["value_spacing_marks"] = round(value_spacing, 4)


#============================================
def _ensure_coordinates(template: dict) -> None:
	"""Add v2 coordinate fields from v3 mark index fields if missing.

	Computes normalized coordinates from fractional timing mark
	indices so that legacy code reading coordinate fields still works.

	Args:
		template: template dict with timing_marks and v3 mark index fields
	"""
	timing = template.get("timing_marks", {})
	left_edge = timing.get("left_edge", {})
	top_edge = timing.get("top_edge", {})
	answers = template["answers"]
	# compute mark step sizes for spacing conversion
	left_step = ((left_edge["end_y"] - left_edge["start_y"])
		/ max(1, left_edge["expected_count"] - 1))
	top_step = ((top_edge["end_x"] - top_edge["start_x"])
		/ max(1, top_edge["expected_count"] - 1))
	# answer columns
	for col_key in ("left_column", "right_column"):
		col = answers[col_key]
		if "first_question_y" in col:
			continue
		# convert left mark index to y-coordinate
		norm_y = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			col["first_row_mark_index"],
			left_edge["start_y"], left_edge["end_y"],
			left_edge["expected_count"])
		# convert mark spacing to y-spacing
		spacing_y = col["row_spacing_marks"] * left_step
		# convert each choice mark index to x-coordinate
		choice_x = {}
		for letter, idx in col["choice_mark_indices"].items():
			norm_x = omr_utils.timing_mark_anchors.mark_index_to_normalized(
				idx,
				top_edge["start_x"], top_edge["end_x"],
				top_edge["expected_count"])
			choice_x[letter] = round(norm_x, 6)
		col["first_question_y"] = round(norm_y, 6)
		col["question_spacing_y"] = round(spacing_y, 6)
		col["choice_x"] = choice_x
	# student ID grid
	grid = template["student_id"]["grid"]
	if "first_digit_x" not in grid:
		# digit x from top mark index
		norm_x = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			grid["first_digit_mark_index"],
			top_edge["start_x"], top_edge["end_x"],
			top_edge["expected_count"])
		digit_spacing_x = grid["digit_spacing_marks"] * top_step
		# value y from left mark index
		norm_y = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			grid["first_value_mark_index"],
			left_edge["start_y"], left_edge["end_y"],
			left_edge["expected_count"])
		value_spacing_y = grid["value_spacing_marks"] * left_step
		grid["first_digit_x"] = round(norm_x, 6)
		grid["digit_spacing_x"] = round(digit_spacing_x, 6)
		grid["first_value_y"] = round(norm_y, 6)
		grid["value_spacing_y"] = round(value_spacing_y, 6)


#============================================
def migrate_template_to_v3(template: dict) -> dict:
	"""Return a v3 template dict with both mark indices and coordinates.

	V3 stores bubble positions as fractional timing mark indices,
	which express the structural relationship between bubbles and
	the timing mark grid. Both mark indices and normalized
	coordinates are kept in memory for backward compatibility.

	Handles migration from v1, v2, or v3 YAML files:
	- v2 YAML (coordinates): computes mark indices from coordinates
	- v3 YAML (mark indices): computes coordinates from mark indices

	Args:
		template: template dictionary (v2 or v3)

	Returns:
		migrated template dictionary with template_version=3
	"""
	migrated = copy.deepcopy(template)
	# ensure v2 fields first (handles v1 migration)
	migrated = migrate_template_to_v2(migrated)
	_validate_v2_shape(migrated)
	timing = migrated.get("timing_marks", {})
	# timing marks required for v3 conversion
	if not timing.get("left_edge") or not timing.get("top_edge"):
		return migrated
	# ensure both representations are present
	_ensure_mark_indices(migrated)
	_ensure_coordinates(migrated)
	migrated["template_version"] = 3
	return migrated


#============================================
def _validate_v2_shape(template: dict) -> None:
	"""Validate v2 bubble shape values are present and sane."""
	answers = template["answers"]
	shape = answers.get("bubble_shape", {})
	aspect_ratio = float(shape["aspect_ratio"])
	if aspect_ratio <= 0:
		raise ValueError("template v2 bubble_shape aspect_ratio must be > 0")


#============================================
def load_template(yaml_path: str) -> dict:
	"""Load and validate a form geometry YAML template.

	Accepts v1, v2, or v3 YAML templates and normalizes to the
	latest version. The returned dict always has both mark index
	fields (v3) and coordinate fields (v2) for backward compatibility.

	Args:
		yaml_path: path to the YAML template file

	Returns:
		parsed template dictionary (normalized to latest version)

	Raises:
		FileNotFoundError: if the YAML file does not exist
		ValueError: if required keys are missing or invalid
	"""
	if not os.path.isfile(yaml_path):
		raise FileNotFoundError(f"template not found: {yaml_path}")
	with open(yaml_path, "r") as fh:
		template = yaml.safe_load(fh)
	if not isinstance(template, dict):
		raise ValueError("template YAML must parse to a dictionary")
	_validate_required_sections(template)
	_validate_answer_columns(template["answers"])
	# migrate through v2 to v3 (handles v1, v2, and v3 inputs)
	template = migrate_template_to_v3(template)
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
	left = answers["left_column"]
	right = answers["right_column"]
	left_range = left["question_range"]
	right_range = right["question_range"]
	if left_range[0] <= question <= left_range[1]:
		col = left
		offset = question - left_range[0]
	elif right_range[0] <= question <= right_range[1]:
		col = right
		offset = question - right_range[0]
	else:
		raise ValueError(f"question {question} not in any column range")
	norm_y = col["first_question_y"] + offset * col["question_spacing_y"]
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
	norm_x = grid["first_digit_x"] + digit * grid["digit_spacing_x"]
	norm_y = grid["first_value_y"] + value * grid["value_spacing_y"]
	return (norm_x, norm_y)


#============================================
def to_pixels(norm_x: float, norm_y: float, width: int, height: int) -> tuple:
	"""Convert normalized coordinates to pixel coordinates."""
	px = int(round(norm_x * width))
	py = int(round(norm_y * height))
	return (px, py)


#============================================
def get_bubble_radius_px(template: dict, width: int, height: int) -> int:
	"""Return bubble radius in pixels for answer bubbles."""
	norm_radius = template["answers"]["bubble_radius"]
	min_dim = min(width, height)
	radius_px = int(round(norm_radius * min_dim))
	return max(radius_px, 3)


#============================================
def get_all_question_coords(template: dict) -> list:
	"""Return all bubble positions for the answer grid."""
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
def get_all_student_id_coords(template: dict) -> list:
	"""Return all bubble positions for the student ID grid."""
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
