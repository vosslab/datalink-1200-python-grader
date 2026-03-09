"""Tests for grade_answers.grade_student blank/multiple reporting."""

# local repo modules
import grade_answers


#============================================
def test_grade_student_tracks_blank_and_multiple() -> None:
	"""Blank and multiple flags should be tracked and scored as incorrect."""
	key_data = {
		"student_id": "KEY",
		"answers": {1: "A", 2: "B", 3: "C", 4: ""},
		"confidences": {},
		"flags": "",
	}
	student_data = {
		"student_id": "S001",
		"answers": {1: "A", 2: "", 3: "C", 4: "D"},
		"confidences": {1: 0.8, 2: 0.0, 3: 0.1, 4: 0.2},
		"flags": "q2:BLANK q3:MULTIPLE(A)",
	}
	graded = grade_answers.grade_student(student_data, key_data)
	assert graded["raw_score"] == 1
	assert graded["total_questions"] == 3
	assert graded["percentage"] == (1 / 3 * 100.0)
	assert graded["per_question"][1] == 1
	assert graded["per_question"][2] == 0
	assert graded["per_question"][3] == 0
	assert graded["per_question"][4] == -1
	assert graded["per_question_status"][1] == "correct"
	assert graded["per_question_status"][2] == "blank"
	assert graded["per_question_status"][3] == "multiple"
	assert graded["per_question_status"][4] == "not_graded"
	assert graded["num_blank"] == 1
	assert graded["num_multiple"] == 1
	assert graded["blank_questions"] == [2]
	assert graded["multiple_questions"] == [3]


#============================================
def test_grade_student_wrong_choice_unchanged_without_flags() -> None:
	"""Unflagged non-matching answers remain wrong_choice."""
	key_data = {
		"student_id": "KEY",
		"answers": {1: "A", 2: "B"},
		"confidences": {},
		"flags": "",
	}
	student_data = {
		"student_id": "S002",
		"answers": {1: "D", 2: "B"},
		"confidences": {1: 0.9, 2: 0.9},
		"flags": "",
	}
	graded = grade_answers.grade_student(student_data, key_data)
	assert graded["raw_score"] == 1
	assert graded["total_questions"] == 2
	assert graded["per_question_status"][1] == "wrong_choice"
	assert graded["per_question_status"][2] == "correct"
	assert graded["num_blank"] == 0
	assert graded["num_multiple"] == 0
