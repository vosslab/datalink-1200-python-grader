"""Refine template coordinates from high-confidence empty-bubble fits."""

# Standard Library
import copy
import os

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader
import omr_utils.image_registration
import omr_utils.bubble_reader


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


#============================================
def collect_image_paths(input_path: str) -> list:
	"""Collect image paths from a file path or directory path."""
	if os.path.isfile(input_path):
		return [input_path]
	if os.path.isdir(input_path):
		paths = []
		for filename in os.listdir(input_path):
			_, ext = os.path.splitext(filename)
			if ext.lower() in IMAGE_EXTENSIONS:
				paths.append(os.path.join(input_path, filename))
		return sorted(paths)
	raise FileNotFoundError(f"input path not found: {input_path}")


#============================================
def collect_empty_offsets(template: dict, results: list,
	width: int, height: int, empty_score_max: float = 0.12) -> list:
	"""Collect template-to-refined offsets from likely empty bubbles."""
	choices = template["answers"]["choices"]
	offsets = []
	for entry in results:
		q_num = int(entry["question"])
		answer = entry.get("answer", "")
		flags = entry.get("flags", "")
		scores = entry.get("scores", {})
		positions = entry.get("positions", {})
		if "MULTIPLE" in flags:
			continue
		for choice in choices:
			# candidate empty zones:
			# 1) all choices in BLANK rows
			# 2) non-selected choices in non-blank rows
			is_candidate = False
			if flags == "BLANK":
				is_candidate = True
			elif answer and choice != answer:
				is_candidate = True
			if not is_candidate:
				continue
			score = float(scores.get(choice, 1.0))
			if score > empty_score_max:
				continue
			if choice not in positions:
				continue
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			tx, ty = omr_utils.template_loader.to_pixels(
				norm_x, norm_y, width, height)
			rx, ry = positions[choice]
			offsets.append({
				"question": q_num,
				"choice": choice,
				"dx": float(rx - tx),
				"dy": float(ry - ty),
				"score": score,
			})
	return offsets


#============================================
def aggregate_offsets(offsets: list, min_samples: int = 2,
	outlier_radius_px: float = 6.0) -> dict:
	"""Aggregate per-bubble offsets with robust median and outlier trimming."""
	grouped = {}
	for entry in offsets:
		key = (int(entry["question"]), str(entry["choice"]))
		grouped.setdefault(key, []).append(entry)
	aggregated = {}
	for key, entries in grouped.items():
		if len(entries) < min_samples:
			continue
		dx_vals = numpy.array([e["dx"] for e in entries], dtype=float)
		dy_vals = numpy.array([e["dy"] for e in entries], dtype=float)
		med_dx = float(numpy.median(dx_vals))
		med_dy = float(numpy.median(dy_vals))
		dists = numpy.sqrt((dx_vals - med_dx) ** 2 + (dy_vals - med_dy) ** 2)
		keep = dists <= float(outlier_radius_px)
		if int(numpy.sum(keep)) < min_samples:
			continue
		trim_dx = dx_vals[keep]
		trim_dy = dy_vals[keep]
		aggregated[key] = {
			"dx": float(numpy.median(trim_dx)),
			"dy": float(numpy.median(trim_dy)),
			"count": int(trim_dx.size),
		}
	return aggregated


#============================================
def _fit_column_updates(template: dict, aggregated: dict,
	width: int, height: int, column_key: str) -> dict:
	"""Fit updated first_y/spacing_y/choice_x for one answer column."""
	answers = template["answers"]
	column = answers[column_key]
	choices = answers["choices"]
	q_start, q_end = column["question_range"]
	# corrected x medians per choice
	choice_x = {}
	for choice in choices:
		x_vals = []
		for q_num in range(q_start, q_end + 1):
			key = (q_num, choice)
			if key not in aggregated:
				continue
			norm_x, _ = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			cx = norm_x + aggregated[key]["dx"] / float(width)
			x_vals.append(cx)
		if x_vals:
			choice_x[choice] = float(numpy.median(x_vals))
		else:
			choice_x[choice] = float(column["choice_x"][choice])
	# corrected y medians per question (across choices)
	row_indices = []
	row_y_vals = []
	for q_num in range(q_start, q_end + 1):
		y_vals = []
		for choice in choices:
			key = (q_num, choice)
			if key not in aggregated:
				continue
			_, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			cy = norm_y + aggregated[key]["dy"] / float(height)
			y_vals.append(cy)
		if y_vals:
			row_indices.append(float(q_num - q_start))
			row_y_vals.append(float(numpy.median(y_vals)))
	if len(row_indices) >= 2:
		slope, intercept = numpy.polyfit(
			numpy.array(row_indices, dtype=float),
			numpy.array(row_y_vals, dtype=float),
			1)
		first_question_y = float(intercept)
		question_spacing_y = float(slope)
	else:
		first_question_y = float(column["first_question_y"])
		question_spacing_y = float(column["question_spacing_y"])
	updates = {
		"first_question_y": round(
			max(0.0, min(1.0, first_question_y)), 6),
		"question_spacing_y": round(
			max(1e-6, min(1.0, question_spacing_y)), 6),
		"choice_x": {},
	}
	for choice in choices:
		updates["choice_x"][choice] = round(
			max(0.0, min(1.0, choice_x[choice])), 6)
	return updates


#============================================
def apply_refined_offsets(template: dict, aggregated: dict,
	width: int, height: int) -> dict:
	"""Return a template with answer columns updated from offsets."""
	refined = copy.deepcopy(template)
	answers = refined["answers"]
	left_updates = _fit_column_updates(
		template, aggregated, width, height, "left_column")
	right_updates = _fit_column_updates(
		template, aggregated, width, height, "right_column")
	answers["left_column"]["first_question_y"] = left_updates["first_question_y"]
	answers["left_column"]["question_spacing_y"] = left_updates["question_spacing_y"]
	answers["left_column"]["choice_x"] = left_updates["choice_x"]
	answers["right_column"]["first_question_y"] = right_updates["first_question_y"]
	answers["right_column"]["question_spacing_y"] = right_updates["question_spacing_y"]
	answers["right_column"]["choice_x"] = right_updates["choice_x"]
	return refined


#============================================
def refine_template_from_images(template: dict, image_paths: list,
	registered: bool = False, empty_score_max: float = 0.12,
	min_samples: int = 2, outlier_radius_px: float = 6.0) -> tuple:
	"""Run extraction on images and compute a refined template.

	Returns:
		(refined_template, report_dict)
	"""
	canon_w = int(template["canonical"]["width_px"])
	canon_h = int(template["canonical"]["height_px"])
	all_offsets = []
	per_image_stats = []
	for image_path in image_paths:
		image = omr_utils.image_registration.load_image(image_path)
		if registered:
			registered_img = image
			if registered_img.shape[1] != canon_w or registered_img.shape[0] != canon_h:
				registered_img = cv2.resize(
					registered_img, (canon_w, canon_h),
					interpolation=cv2.INTER_AREA)
		else:
			registered_img = omr_utils.image_registration.register_image(
				image, canon_w, canon_h)
		results = omr_utils.bubble_reader.read_answers(registered_img, template)
		offsets = collect_empty_offsets(
			template, results, canon_w, canon_h, empty_score_max)
		all_offsets.extend(offsets)
		per_image_stats.append({
			"image": image_path,
			"offset_count": len(offsets),
			"answered": int(sum(1 for r in results if r["answer"])),
		})
	aggregated = aggregate_offsets(
		all_offsets,
		min_samples=min_samples,
		outlier_radius_px=outlier_radius_px)
	refined_template = apply_refined_offsets(
		template, aggregated, canon_w, canon_h)
	report = {
		"image_count": len(image_paths),
		"raw_offset_count": len(all_offsets),
		"aggregated_bubbles": len(aggregated),
		"per_image": per_image_stats,
	}
	return (refined_template, report)
