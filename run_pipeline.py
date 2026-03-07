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
import omr_utils.bubble_template_extractor
import omr_utils.debug_drawing
import omr_utils.timing_mark_anchors
import omr_utils.student_id_reader
import omr_utils.csv_writer
import omr_utils.xlsx_writer
import omr_utils.slot_map
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
	parser.add_argument(
		'-r', '--refine-mode', dest='refine_mode',
		type=str, default='ncc',
		choices=('lattice', 'ncc'),
		help="Refinement mode (default: ncc)"
	)
	parser.add_argument(
		'--ncc-diag', dest='ncc_diag', action='store_true',
		help="Write per-slot NCC diagnostic CSV to output directory"
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
	output_dir: str, debug: bool = False,
	refine_mode: str = "ncc",
	ncc_diag: bool = False) -> dict:
	"""Process a single scantron image through registration and extraction.

	Args:
		image_path: path to the raw scan or photo
		template: loaded template dictionary
		output_dir: directory for output files
		debug: whether to save debug overlays
		refine_mode: refinement mode
		ncc_diag: whether to write NCC diagnostic CSV

	Returns:
		dict with keys: csv_path, student_id, num_answered
	"""
	base_name = os.path.splitext(os.path.basename(image_path))[0]
	# load and register the raw image
	raw_image = omr_utils.image_registration.load_image(image_path)
	registered = omr_utils.image_registration.register_image(raw_image)
	# compute timing mark transform for anchor-derived measure_cfgetry
	gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	# normalize full image contrast (1% black / 25% white stretch)
	gray = omr_utils.bubble_template_extractor.normalize_roi_percentile(gray)
	raw_transform = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	print(f"  anchor confidence: left={raw_transform.get('left_confidence', 0):.3f}"
		f" top={raw_transform.get('top_confidence', 0):.3f}"
		f" marks: left={len(raw_transform.get('left_marks', []))}"
		f" top={len(raw_transform.get('top_marks', []))}")
	# build SlotMap from timing mark transform (single measure_cfgetry authority)
	slot_map = omr_utils.slot_map.SlotMap(raw_transform, template)
	measure_cfg = slot_map.measure_cfg()
	# read student ID using SlotMap lattice geometry
	student_id = omr_utils.student_id_reader.read_student_id(
		registered, template, slot_map)
	# build NCC diagnostic CSV path if requested
	ncc_diag_path = None
	if ncc_diag:
		ncc_diag_path = os.path.join(output_dir,
			f"{base_name}_ncc_diagnostics.csv")
	# read answer bubbles using SlotMap
	answers, ncc_diag_data = omr_utils.bubble_reader.read_answers(
		registered, template, slot_map=slot_map,
		refine_mode=refine_mode,
		ncc_diag_path=ncc_diag_path)
	# count non-blank answers (exclude MULTIPLE-flagged as unreliable)
	num_answered = sum(
		1 for a in answers
		if a["answer"] and "MULTIPLE" not in a.get("flags", ""))
	# write answers CSV
	csv_path = os.path.join(output_dir, f"{base_name}_answers.csv")
	omr_utils.csv_writer.write_answers_csv(csv_path, student_id, answers)
	# save debug images
	if debug:
		# scored: bubble outlines with filled/not determination and confidence
		scored_img = omr_utils.debug_drawing.draw_scored_overlay(
			registered, template, answers, measure_cfg, slot_map=slot_map)
		scored_path = os.path.join(output_dir, f"{base_name}_scored.png")
		cv2.imwrite(scored_path, scored_img)
		print(f"    scored: {scored_path}")
		# lattice crosshairs: verify measure_cfgetry independently of measurement
		lattice_img = omr_utils.debug_drawing.draw_lattice_crosshairs(
			registered, slot_map, template)
		lattice_path = os.path.join(output_dir, f"{base_name}_lattice.png")
		cv2.imwrite(lattice_path, lattice_img)
		print(f"    lattice: {lattice_path}")
		# debug: timing marks + guide lines + bubble overlays combined
		debug_img = omr_utils.debug_drawing.draw_combined_debug(
			registered, template, raw_transform, answers, measure_cfg,
			slot_map=slot_map)
		debug_path = os.path.join(output_dir, f"{base_name}_debug.png")
		cv2.imwrite(debug_path, debug_img)
		print(f"    debug: {debug_path}")
		# NCC shift overlay: triple-dot seed/NCC/final positions
		if ncc_diag_data:
			ncc_img = omr_utils.debug_drawing.draw_ncc_shift_overlay(
				registered, answers)
			ncc_path = os.path.join(output_dir,
				f"{base_name}_ncc_shifts.png")
			cv2.imwrite(ncc_path, ncc_img)
			print(f"    ncc_shifts: {ncc_path}")
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
		args.key_file, template, args.output_dir, args.debug,
		refine_mode=args.refine_mode,
		ncc_diag=args.ncc_diag)
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
			image_path, template, args.output_dir, args.debug,
			refine_mode=args.refine_mode,
			ncc_diag=args.ncc_diag)
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
