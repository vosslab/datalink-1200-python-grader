"""Load and query DataLink 1200 form geometry from a YAML template."""

# Standard Library
import os

# PIP3 modules
import yaml


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
def migrate_template_to_v2(template: dict) -> dict:
	"""Validate and normalize a template dict to v2.

	The YAML is already v2; this function validates required
	sections and sets template_version=2. Legacy v1 bubble_geometry
	is removed if present.

	Args:
		template: raw template dictionary loaded from YAML

	Returns:
		validated template dictionary with template_version=2
	"""
	_validate_required_sections(template)
	answers = template["answers"]
	_validate_answer_columns(answers)
	# remove legacy v1 bubble_geometry if still present
	if "bubble_geometry" in answers:
		del answers["bubble_geometry"]
	template["template_version"] = 2
	return template


#============================================
def load_template(yaml_path: str) -> dict:
	"""Load and validate a form geometry YAML template.

	Validates required sections and normalizes to v2.
	Bubble positions use local lattice column indices in
	choice_columns; no 53-grid or mark index conversion.

	Args:
		yaml_path: path to the YAML template file

	Returns:
		parsed template dictionary (normalized to v2)

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
	template = migrate_template_to_v2(template)
	return template



