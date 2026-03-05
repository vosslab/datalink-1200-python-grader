"""End-to-end smoke tests for the full OMR pipeline."""

# Standard Library
import os

# PIP3 modules
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader
import omr_utils.student_id_reader
import omr_utils.csv_writer
import grade_answers

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output_smoke")

KEY_SHEET = os.path.join(SCANTRON_DIR,
	"8B5D0C61-395B-4E28-AED1-3E0D9959FAE0_result.jpg")
BOTELLO = os.path.join(SCANTRON_DIR, "Botello_2.24.26.jpg")


#============================================
def _skip_if_no_scantrons() -> None:
	"""Skip test if scantron images are not available."""
	if not os.path.isdir(SCANTRON_DIR):
		pytest.skip("scantrons/ directory not found")


#============================================
@pytest.fixture(scope="module")
def template() -> dict:
	"""Load template once for all tests in this module."""
	loaded = omr_utils.template_loader.load_template(TEMPLATE_PATH)
	return loaded


#============================================
@pytest.fixture(scope="module")
def setup_output() -> str:
	"""Create smoke test output directory."""
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	return OUTPUT_DIR


#============================================
class TestEndToEnd:
	"""End-to-end pipeline smoke tests."""

	def test_key_extraction_q28_e(self, template: dict,
		setup_output: str) -> None:
		"""Answer key Q28 should be E."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_SHEET)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		q28 = [r for r in results if r["question"] == 28][0]
		assert q28["answer"] == "E"

	def test_botello_extraction_q28_e(self, template: dict,
		setup_output: str) -> None:
		"""Botello Q28 should be E."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(BOTELLO)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		q28 = [r for r in results if r["question"] == 28][0]
		assert q28["answer"] == "E"

	def test_key_has_at_least_50_answers(self, template: dict,
		setup_output: str) -> None:
		"""Answer key should have at least 50 detected answers."""
		_skip_if_no_scantrons()
		img = omr_utils.image_registration.load_image(KEY_SHEET)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		answered = sum(1 for r in results if r["answer"])
		assert answered >= 50

	def test_grading_produces_valid_score(self, template: dict,
		setup_output: str) -> None:
		"""Grading Botello against key produces a valid percentage."""
		_skip_if_no_scantrons()
		# process key
		img_key = omr_utils.image_registration.load_image(KEY_SHEET)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		reg_key = omr_utils.image_registration.register_image(
			img_key, canon_w, canon_h)
		key_results = omr_utils.bubble_reader.read_answers(reg_key, template)
		key_id = omr_utils.student_id_reader.read_student_id(reg_key, template)
		key_csv = os.path.join(setup_output, "key_answers.csv")
		omr_utils.csv_writer.write_answers_csv(key_csv, key_id, key_results)
		# process Botello
		img_bot = omr_utils.image_registration.load_image(BOTELLO)
		reg_bot = omr_utils.image_registration.register_image(
			img_bot, canon_w, canon_h)
		bot_results = omr_utils.bubble_reader.read_answers(reg_bot, template)
		bot_id = omr_utils.student_id_reader.read_student_id(reg_bot, template)
		bot_csv = os.path.join(setup_output, "botello_answers.csv")
		omr_utils.csv_writer.write_answers_csv(bot_csv, bot_id, bot_results)
		# grade
		key_data = omr_utils.csv_writer.read_answers_csv(key_csv)
		bot_data = omr_utils.csv_writer.read_answers_csv(bot_csv)
		graded = grade_answers.grade_student(bot_data, key_data)
		assert 0.0 <= graded["percentage"] <= 100.0
		assert graded["total_questions"] > 0
		assert graded["raw_score"] >= 0
