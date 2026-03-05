"""Write answer extraction results to CSV files."""

# Standard Library
import csv


#============================================
def write_answers_csv(output_path: str, student_id: str,
	results: list) -> None:
	"""Write extracted answers to a CSV file.

	Format: student_id,q1,q2,...,q100,conf1,conf2,...,conf100,flags
	Confidence is the gap between the top score and second-best score,
	normalized so higher = more confident in the detected answer.

	Args:
		output_path: path for the output CSV file
		student_id: student ID string
		results: list of answer dicts from bubble_reader.read_answers
	"""
	# collect flags
	flag_parts = []
	for entry in results:
		if entry["flags"]:
			flag_parts.append(f"q{entry['question']}:{entry['flags']}")
	flags_str = " ".join(flag_parts)
	# build row: student_id, answers, confidences, flags
	row = [student_id]
	sorted_results = sorted(results, key=lambda e: e["question"])
	# add answer columns
	for entry in sorted_results:
		row.append(entry["answer"])
	# add confidence columns
	for entry in sorted_results:
		confidence = _compute_confidence(entry["scores"])
		row.append(f"{confidence:.3f}")
	row.append(flags_str)
	# write CSV
	with open(output_path, "w", newline="") as fh:
		writer = csv.writer(fh)
		num_q = len(results)
		header = ["student_id"]
		for i in range(1, num_q + 1):
			header.append(f"q{i}")
		for i in range(1, num_q + 1):
			header.append(f"conf{i}")
		header.append("flags")
		writer.writerow(header)
		writer.writerow(row)


#============================================
def _compute_confidence(scores: dict) -> float:
	"""Compute confidence for an answer based on score gap.

	Confidence is the gap between the top score and the second-best
	score. Higher confidence means the top choice stands out clearly.

	Args:
		scores: dict mapping choice letter to fill score

	Returns:
		confidence value (0.0 to ~0.5, higher is better)
	"""
	if not scores:
		return 0.0
	sorted_scores = sorted(scores.values(), reverse=True)
	if len(sorted_scores) < 2:
		return sorted_scores[0]
	gap = sorted_scores[0] - sorted_scores[1]
	return gap


#============================================
def read_answers_csv(csv_path: str) -> dict:
	"""Read an answers CSV file back into a dictionary.

	Args:
		csv_path: path to the CSV file

	Returns:
		dict with keys: student_id (str), answers (dict of int->str),
		confidences (dict of int->float), flags (str)
	"""
	with open(csv_path, "r") as fh:
		reader = csv.DictReader(fh)
		row = next(reader)
	student_id = row["student_id"]
	answers = {}
	confidences = {}
	for key, value in row.items():
		if key.startswith("q") and key[1:].isdigit():
			q_num = int(key[1:])
			answers[q_num] = value
		elif key.startswith("conf") and key[4:].isdigit():
			q_num = int(key[4:])
			confidences[q_num] = float(value)
	flags = row.get("flags", "")
	result = {
		"student_id": student_id,
		"answers": answers,
		"confidences": confidences,
		"flags": flags,
	}
	return result
