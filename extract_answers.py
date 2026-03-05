#!/usr/bin/env python3
"""Extract answers from a registered scantron image."""

# Standard Library
import os
import argparse

# PIP3 modules
import cv2

# local repo modules
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader
import omr_utils.student_id_reader
import omr_utils.csv_writer


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		parsed argument namespace
	"""
	parser = argparse.ArgumentParser(
		description="Extract answers from a registered scantron image"
	)
	parser.add_argument(
		'-i', '--input', dest='input_file', required=True,
		help="Path to registered image (or raw scan, will auto-register)"
	)
	parser.add_argument(
		'-o', '--output', dest='output_file', required=True,
		help="Path for output answers CSV"
	)
	parser.add_argument(
		'-t', '--template', dest='template_file',
		default=None,
		help="Path to template YAML (default: config/dl1200_template.yaml)"
	)
	parser.add_argument(
		'-r', '--registered', dest='is_registered', action='store_true',
		help="Input is already registered (skip registration step)"
	)
	parser.add_argument(
		'-d', '--debug', dest='debug', action='store_true',
		help="Save debug overlay image alongside output"
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
def main() -> None:
	"""Main entry point for answer extraction."""
	args = parse_args()
	# resolve template path
	template_path = args.template_file
	if template_path is None:
		template_path = get_default_template_path()
	# load template
	template = omr_utils.template_loader.load_template(template_path)
	canon_w = template["canonical"]["width_px"]
	canon_h = template["canonical"]["height_px"]
	# load and optionally register image
	print(f"loading: {args.input_file}")
	image = omr_utils.image_registration.load_image(args.input_file)
	if args.is_registered:
		registered = image
		# resize to canonical if needed
		if registered.shape[1] != canon_w or registered.shape[0] != canon_h:
			registered = cv2.resize(registered, (canon_w, canon_h),
				interpolation=cv2.INTER_AREA)
	else:
		# auto-register
		print("  registering image...")
		registered = omr_utils.image_registration.register_image(
			image, canon_w, canon_h)
	print(f"  image size: {registered.shape[1]}x{registered.shape[0]}")
	# read student ID
	student_id = omr_utils.student_id_reader.read_student_id(registered, template)
	print(f"  student ID: {student_id}")
	# read answers
	results = omr_utils.bubble_reader.read_answers(registered, template)
	# count answers
	answered = sum(1 for r in results if r["answer"])
	blank = sum(1 for r in results if r["flags"] == "BLANK")
	multiple = sum(1 for r in results if "MULTIPLE" in r["flags"])
	print(f"  answered: {answered}, blank: {blank}, multiple: {multiple}")
	# ensure output directory exists
	output_dir = os.path.dirname(args.output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	# write CSV
	omr_utils.csv_writer.write_answers_csv(args.output_file, student_id, results)
	print(f"  output: {args.output_file}")
	# debug overlay
	if args.debug:
		debug_img = omr_utils.bubble_reader.draw_answer_debug(
			registered, template, results)
		debug_path = args.output_file.replace(".csv", "_debug.png")
		cv2.imwrite(debug_path, debug_img)
		print(f"  debug: {debug_path}")


#============================================
if __name__ == '__main__':
	main()
