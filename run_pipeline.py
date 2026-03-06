#!/usr/bin/env python3
"""Batch pipeline: register, extract, and grade scantron images."""

# Standard Library
import os
import glob
import argparse

# PIP3 modules
import cv2

# local repo modules
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader
import omr_utils.student_id_reader
import omr_utils.csv_writer
import omr_utils.xlsx_writer
import grade_answers


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		parsed argument namespace
	"""
	parser = argparse.ArgumentParser(
		description="Batch process scantron images: register, extract, and grade"
	)
	parser.add_argument(
		'-i', '--input', dest='input_path', required=True,
		help="Path to scan image or directory of images"
	)
	parser.add_argument(
		'-k', '--key', dest='key_file', required=True,
		help="Path to answer key image (will be processed first)"
	)
	parser.add_argument(
		'-o', '--output-dir', dest='output_dir',
		default="data/output",
		help="Output directory (default: data/output)"
	)
	parser.add_argument(
		'-t', '--template', dest='template_file',
		default=None,
		help="Path to template YAML (default: config/dl1200_template.yaml)"
	)
	parser.add_argument(
		'-d', '--debug', dest='debug', action='store_true',
		help="Enable debug overlays for all stages"
	)
	args = parser.parse_args()
	return args


#============================================
def get_default_template_path() -> str:
	"""Return the default template path relative to this script.

	Returns:
		absolute path to config/dl1200_template.yaml
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	template_path = os.path.join(script_dir, "config", "dl1200_template.yaml")
	return template_path


#============================================
def collect_image_paths(input_path: str) -> list:
	"""Collect image file paths from a path (file or directory).

	Args:
		input_path: path to single image or directory

	Returns:
		list of image file paths
	"""
	image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
	if os.path.isfile(input_path):
		return [input_path]
	if os.path.isdir(input_path):
		paths = []
		for ext in image_extensions:
			pattern = os.path.join(input_path, f"*{ext}")
			paths.extend(glob.glob(pattern))
			# also uppercase extensions
			pattern_upper = os.path.join(input_path, f"*{ext.upper()}")
			paths.extend(glob.glob(pattern_upper))
		# deduplicate and sort
		paths = sorted(set(paths))
		return paths
	raise FileNotFoundError(f"input path not found: {input_path}")


#============================================
def process_single_image(image_path: str, template: dict,
	output_dir: str, debug: bool = False) -> dict:
	"""Process a single scantron image through registration and extraction.

	Args:
		image_path: path to the raw scan or photo
		template: loaded template dictionary
		output_dir: directory for output files
		debug: whether to save debug overlays

	Returns:
		dict with keys: csv_path, student_id, num_answered
	"""
	canon_w = template["canonical"]["width_px"]
	canon_h = template["canonical"]["height_px"]
	# load and register
	image = omr_utils.image_registration.load_image(image_path)
	registered = omr_utils.image_registration.register_image(
		image, canon_w, canon_h)
	# base filename for outputs
	base_name = os.path.splitext(os.path.basename(image_path))[0]
	# save registered image
	reg_path = os.path.join(output_dir, f"{base_name}_registered.png")
	cv2.imwrite(reg_path, registered)
	# extract student ID
	student_id = omr_utils.student_id_reader.read_student_id(registered, template)
	# extract answers
	results = omr_utils.bubble_reader.read_answers(registered, template)
	# write CSV
	csv_path = os.path.join(output_dir, f"{base_name}_answers.csv")
	omr_utils.csv_writer.write_answers_csv(csv_path, student_id, results)
	# debug overlay
	if debug:
		debug_img = omr_utils.bubble_reader.draw_answer_debug(
			registered, template, results)
		debug_path = os.path.join(output_dir, f"{base_name}_debug.png")
		cv2.imwrite(debug_path, debug_img)
	num_answered = sum(1 for r in results if r["answer"])
	result = {
		"csv_path": csv_path,
		"student_id": student_id,
		"num_answered": num_answered,
	}
	return result


#============================================
def main() -> None:
	"""Main entry point for batch pipeline."""
	args = parse_args()
	# resolve template path
	template_path = args.template_file
	if template_path is None:
		template_path = get_default_template_path()
	template = omr_utils.template_loader.load_template(template_path)
	# ensure output directory exists
	os.makedirs(args.output_dir, exist_ok=True)
	# process answer key first
	print("=== Processing answer key ===")
	print(f"  {args.key_file}")
	key_result = process_single_image(
		args.key_file, template, args.output_dir, args.debug)
	key_csv = key_result["csv_path"]
	print(f"  student ID: {key_result['student_id']}")
	print(f"  answers: {key_result['num_answered']}")
	print(f"  csv: {key_csv}")
	# collect student images
	image_paths = collect_image_paths(args.input_path)
	# exclude the key image from student list
	key_abs = os.path.abspath(args.key_file)
	student_paths = [p for p in image_paths if os.path.abspath(p) != key_abs]
	if not student_paths:
		print("no student images found")
		return
	# load key CSV for grading
	key_data = omr_utils.csv_writer.read_answers_csv(key_csv)
	# process each student image
	# collect results for XLSX summary
	all_student_data = []
	all_graded_results = []
	print(f"\n=== Processing {len(student_paths)} student images ===")
	for image_path in student_paths:
		base_name = os.path.splitext(os.path.basename(image_path))[0]
		print(f"\n  {base_name}")
		student_result = process_single_image(
			image_path, template, args.output_dir, args.debug)
		print(f"    ID: {student_result['student_id']}")
		print(f"    answered: {student_result['num_answered']}")
		# grade against key
		student_data = omr_utils.csv_writer.read_answers_csv(
			student_result["csv_path"])
		graded = grade_answers.grade_student(student_data, key_data)
		grade_csv = os.path.join(args.output_dir, f"{base_name}_grades.csv")
		grade_answers.write_graded_csv(grade_csv, graded, student_data, key_data)
		print(f"    score: {graded['raw_score']}/{graded['total_questions']}"
			f" ({graded['percentage']:.1f}%)")
		if graded["low_confidence"]:
			lc_str = ", ".join(f"q{q}" for q in graded["low_confidence"])
			print(f"    low confidence: {lc_str}")
		all_student_data.append(student_data)
		all_graded_results.append(graded)
	# write XLSX scoring summary
	if all_graded_results:
		xlsx_path = os.path.join(args.output_dir, "scoring_summary.xlsx")
		omr_utils.xlsx_writer.write_scoring_summary(
			xlsx_path, key_data, all_student_data, all_graded_results)
		print(f"\n  XLSX summary: {xlsx_path}")
	print("\n=== Done ===")


#============================================
if __name__ == '__main__':
	main()
