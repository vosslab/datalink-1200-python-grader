#!/usr/bin/env python3
"""Compare student answers to an answer key and produce graded results."""

# Standard Library
import os
import csv
import argparse

# local repo modules
import omr_utils.csv_writer


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		parsed argument namespace
	"""
	parser = argparse.ArgumentParser(
		description="Grade student answers against an answer key"
	)
	parser.add_argument(
		'-i', '--input', dest='input_file', required=True,
		help="Path to student answers CSV"
	)
	parser.add_argument(
		'-k', '--key', dest='key_file', required=True,
		help="Path to answer key CSV (from extract_answers.py)"
	)
	parser.add_argument(
		'-o', '--output', dest='output_file', required=True,
		help="Path for graded results CSV"
	)
	args = parser.parse_args()
	return args


#============================================
def grade_student(student_data: dict, key_data: dict) -> dict:
	"""Grade a student's answers against the answer key.

	Questions where both student and key are blank are not counted.
	Questions where the key has an answer but student is blank count
	as incorrect.

	Args:
		student_data: dict from read_answers_csv (student)
		key_data: dict from read_answers_csv (answer key)

	Returns:
		dict with keys: student_id, raw_score, total_questions,
		percentage, per_question (list of 0/1/-1), low_confidence (list)
	"""
	student_answers = student_data["answers"]
	key_answers = key_data["answers"]
	student_conf = student_data.get("confidences", {})
	correct = 0
	total = 0
	per_question = {}
	low_confidence = []
	for q_num in sorted(key_answers.keys()):
		key_answer = key_answers.get(q_num, "")
		student_answer = student_answers.get(q_num, "")
		# skip questions where key is blank (not part of exam)
		if not key_answer:
			per_question[q_num] = -1
			continue
		total += 1
		if student_answer == key_answer:
			correct += 1
			per_question[q_num] = 1
		else:
			per_question[q_num] = 0
		# track low confidence answers
		conf = student_conf.get(q_num, 0.0)
		if conf < 0.05 and student_answer:
			low_confidence.append(q_num)
	percentage = (correct / total * 100.0) if total > 0 else 0.0
	result = {
		"student_id": student_data["student_id"],
		"raw_score": correct,
		"total_questions": total,
		"percentage": percentage,
		"per_question": per_question,
		"low_confidence": low_confidence,
	}
	return result


#============================================
def write_graded_csv(output_path: str, graded: dict,
	student_data: dict, key_data: dict) -> None:
	"""Write graded results to a CSV file.

	Format: student_id,raw_score,total,percentage,q1,q2,...,q100,flags

	Args:
		output_path: path for the output CSV file
		graded: dict from grade_student
		student_data: original student answers dict
		key_data: original key answers dict
	"""
	# build header
	num_q = len(key_data["answers"])
	header = ["student_id", "raw_score", "total", "percentage"]
	for i in range(1, num_q + 1):
		header.append(f"q{i}")
	header.append("low_confidence")
	header.append("flags")
	# build row
	row = [
		graded["student_id"],
		graded["raw_score"],
		graded["total_questions"],
		f"{graded['percentage']:.1f}",
	]
	for i in range(1, num_q + 1):
		score = graded["per_question"].get(i, -1)
		if score == 1:
			row.append("1")
		elif score == 0:
			row.append("0")
		else:
			row.append("")
	# low confidence questions
	lc_str = " ".join(f"q{q}" for q in graded["low_confidence"])
	row.append(lc_str)
	# flags from original extraction
	row.append(student_data.get("flags", ""))
	# ensure output directory exists
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	# write
	with open(output_path, "w", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(header)
		writer.writerow(row)


#============================================
def main() -> None:
	"""Main entry point for grading."""
	args = parse_args()
	# load student and key CSVs
	print(f"loading key: {args.key_file}")
	key_data = omr_utils.csv_writer.read_answers_csv(args.key_file)
	print(f"loading student: {args.input_file}")
	student_data = omr_utils.csv_writer.read_answers_csv(args.input_file)
	print(f"  student ID: {student_data['student_id']}")
	# grade
	graded = grade_student(student_data, key_data)
	print(f"  score: {graded['raw_score']}/{graded['total_questions']}"
		f" ({graded['percentage']:.1f}%)")
	if graded["low_confidence"]:
		lc_str = ", ".join(f"q{q}" for q in graded["low_confidence"])
		print(f"  low confidence: {lc_str}")
	# write output
	write_graded_csv(args.output_file, graded, student_data, key_data)
	print(f"  output: {args.output_file}")


#============================================
if __name__ == '__main__':
	main()
