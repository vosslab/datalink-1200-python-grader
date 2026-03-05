"""Tests for grade_answers module."""

# Standard Library
import os
import tempfile

# local repo modules
import git_file_utils
import omr_utils.csv_writer
import grade_answers

REPO_ROOT = git_file_utils.get_repo_root()


#============================================
class TestGradeStudent:
	"""Tests for grade_student function."""

	def test_perfect_score(self) -> None:
		"""Perfect answers get 100%."""
		key_data = {
			"student_id": "000000000",
			"answers": {1: "A", 2: "B", 3: "C"},
			"confidences": {},
			"flags": "",
		}
		student_data = {
			"student_id": "123456789",
			"answers": {1: "A", 2: "B", 3: "C"},
			"confidences": {1: 0.200, 2: 0.150, 3: 0.180},
			"flags": "",
		}
		graded = grade_answers.grade_student(student_data, key_data)
		assert graded["raw_score"] == 3
		assert graded["total_questions"] == 3
		assert graded["percentage"] == 100.0

	def test_zero_score(self) -> None:
		"""All wrong answers get 0%."""
		key_data = {
			"student_id": "000000000",
			"answers": {1: "A", 2: "B", 3: "C"},
			"confidences": {},
			"flags": "",
		}
		student_data = {
			"student_id": "123456789",
			"answers": {1: "B", 2: "C", 3: "D"},
			"confidences": {1: 0.200, 2: 0.150, 3: 0.180},
			"flags": "",
		}
		graded = grade_answers.grade_student(student_data, key_data)
		assert graded["raw_score"] == 0
		assert graded["percentage"] == 0.0

	def test_blank_key_excluded(self) -> None:
		"""Questions where key is blank are not counted."""
		key_data = {
			"student_id": "000000000",
			"answers": {1: "A", 2: "", 3: "C"},
			"confidences": {},
			"flags": "",
		}
		student_data = {
			"student_id": "123456789",
			"answers": {1: "A", 2: "B", 3: "C"},
			"confidences": {},
			"flags": "",
		}
		graded = grade_answers.grade_student(student_data, key_data)
		# only Q1 and Q3 counted (key blank for Q2)
		assert graded["total_questions"] == 2
		assert graded["raw_score"] == 2

	def test_blank_student_is_wrong(self) -> None:
		"""Student blank on a keyed question counts as incorrect."""
		key_data = {
			"student_id": "000000000",
			"answers": {1: "A", 2: "B"},
			"confidences": {},
			"flags": "",
		}
		student_data = {
			"student_id": "123456789",
			"answers": {1: "A", 2: ""},
			"confidences": {},
			"flags": "",
		}
		graded = grade_answers.grade_student(student_data, key_data)
		assert graded["raw_score"] == 1
		assert graded["total_questions"] == 2

	def test_low_confidence_flagged(self) -> None:
		"""Low confidence answers are tracked."""
		key_data = {
			"student_id": "000000000",
			"answers": {1: "A", 2: "B"},
			"confidences": {},
			"flags": "",
		}
		student_data = {
			"student_id": "123456789",
			"answers": {1: "A", 2: "B"},
			"confidences": {1: 0.200, 2: 0.020},
			"flags": "",
		}
		graded = grade_answers.grade_student(student_data, key_data)
		# Q2 has low confidence (0.020 < 0.05)
		assert 2 in graded["low_confidence"]
		assert 1 not in graded["low_confidence"]


#============================================
class TestCsvRoundTrip:
	"""Tests for CSV write and read round trip."""

	def test_write_and_read(self) -> None:
		"""Written CSV can be read back correctly."""
		results = [
			{"question": 1, "answer": "A", "scores": {"A": 0.4, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1}, "flags": ""},
			{"question": 2, "answer": "B", "scores": {"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.1, "E": 0.1}, "flags": ""},
			{"question": 3, "answer": "", "scores": {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1}, "flags": "BLANK"},
		]
		with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
			tmp_path = fh.name
		omr_utils.csv_writer.write_answers_csv(tmp_path, "123456789", results)
		data = omr_utils.csv_writer.read_answers_csv(tmp_path)
		assert data["student_id"] == "123456789"
		assert data["answers"][1] == "A"
		assert data["answers"][2] == "B"
		assert data["answers"][3] == ""
		# confidence values should be present
		assert 1 in data["confidences"]
		assert data["confidences"][1] > 0.0
		os.unlink(tmp_path)
