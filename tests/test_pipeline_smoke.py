"""End-to-end smoke tests for the full OMR pipeline."""

# Standard Library
import os
import glob

# PIP3 modules
import pytest

# local repo modules
import git_file_utils
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader
import omr_utils.student_id_reader
import omr_utils.csv_writer
import omr_utils.xlsx_writer
import grade_answers

REPO_ROOT = git_file_utils.get_repo_root()
TEMPLATE_PATH = os.path.join(REPO_ROOT, "config", "dl1200_template.yaml")
SCANTRON_DIR = os.path.join(REPO_ROOT, "scantrons")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output_smoke")

KEY_IMAGE = os.path.join(SCANTRON_DIR,
	"43F257A7-A03D-4CB2-8D7B-3EE057B41FAC_result.jpg")


#============================================
def _skip_if_no_scantrons() -> None:
	"""Skip test if scantron images are not available."""
	if not os.path.isdir(SCANTRON_DIR):
		pytest.skip("scantrons/ directory not found")


#============================================
def _get_student_images() -> list:
	"""Discover student images by excluding the key image.

	Returns:
		sorted list of student image paths
	"""
	key_base = os.path.basename(KEY_IMAGE)
	all_images = sorted(glob.glob(os.path.join(SCANTRON_DIR, "*.jpg")))
	students = [p for p in all_images if os.path.basename(p) != key_base]
	return students


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
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		q28 = [r for r in results if r["question"] == 28][0]
		assert q28["answer"] == "E"

	def test_student_extraction_q28_e(self, template: dict,
		setup_output: str) -> None:
		"""First discovered student Q28 should be E."""
		_skip_if_no_scantrons()
		students = _get_student_images()
		if not students:
			pytest.skip("no student images found")
		img = omr_utils.image_registration.load_image(students[0])
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
		img = omr_utils.image_registration.load_image(KEY_IMAGE)
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		registered = omr_utils.image_registration.register_image(
			img, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered, template)
		answered = sum(1 for r in results if r["answer"])
		assert answered >= 50

	def test_grading_produces_valid_score(self, template: dict,
		setup_output: str) -> None:
		"""Grading first student against key produces a valid percentage."""
		_skip_if_no_scantrons()
		students = _get_student_images()
		if not students:
			pytest.skip("no student images found")
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		# process key
		img_key = omr_utils.image_registration.load_image(KEY_IMAGE)
		reg_key = omr_utils.image_registration.register_image(
			img_key, canon_w, canon_h)
		key_results = omr_utils.bubble_reader.read_answers(reg_key, template)
		key_id = omr_utils.student_id_reader.read_student_id(
			reg_key, template)
		key_csv = os.path.join(setup_output, "key_answers.csv")
		omr_utils.csv_writer.write_answers_csv(key_csv, key_id, key_results)
		# process first student
		img_stu = omr_utils.image_registration.load_image(students[0])
		reg_stu = omr_utils.image_registration.register_image(
			img_stu, canon_w, canon_h)
		stu_results = omr_utils.bubble_reader.read_answers(reg_stu, template)
		stu_id = omr_utils.student_id_reader.read_student_id(
			reg_stu, template)
		stu_csv = os.path.join(setup_output, "student_answers.csv")
		omr_utils.csv_writer.write_answers_csv(
			stu_csv, stu_id, stu_results)
		# grade
		key_data = omr_utils.csv_writer.read_answers_csv(key_csv)
		stu_data = omr_utils.csv_writer.read_answers_csv(stu_csv)
		graded = grade_answers.grade_student(stu_data, key_data)
		assert 0.0 <= graded["percentage"] <= 100.0
		assert graded["total_questions"] > 0
		assert graded["raw_score"] >= 0

	def test_xlsx_created(self, template: dict,
		setup_output: str) -> None:
		"""Pipeline produces scoring_summary.xlsx in output dir."""
		_skip_if_no_scantrons()
		students = _get_student_images()
		if not students:
			pytest.skip("no student images found")
		canon_w = template["canonical"]["width_px"]
		canon_h = template["canonical"]["height_px"]
		# process key
		img_key = omr_utils.image_registration.load_image(KEY_IMAGE)
		reg_key = omr_utils.image_registration.register_image(
			img_key, canon_w, canon_h)
		key_results = omr_utils.bubble_reader.read_answers(reg_key, template)
		key_id = omr_utils.student_id_reader.read_student_id(
			reg_key, template)
		key_csv = os.path.join(setup_output, "key_answers.csv")
		omr_utils.csv_writer.write_answers_csv(key_csv, key_id, key_results)
		key_data = omr_utils.csv_writer.read_answers_csv(key_csv)
		# process first student
		img_stu = omr_utils.image_registration.load_image(students[0])
		reg_stu = omr_utils.image_registration.register_image(
			img_stu, canon_w, canon_h)
		stu_results = omr_utils.bubble_reader.read_answers(reg_stu, template)
		stu_id = omr_utils.student_id_reader.read_student_id(
			reg_stu, template)
		stu_csv = os.path.join(setup_output, "student_answers.csv")
		omr_utils.csv_writer.write_answers_csv(
			stu_csv, stu_id, stu_results)
		stu_data = omr_utils.csv_writer.read_answers_csv(stu_csv)
		# grade
		graded = grade_answers.grade_student(stu_data, key_data)
		# write XLSX
		xlsx_path = os.path.join(setup_output, "scoring_summary.xlsx")
		omr_utils.xlsx_writer.write_scoring_summary(
			xlsx_path, key_data, [stu_data], [graded])
		assert os.path.isfile(xlsx_path)
