"""Debug overlay drawing for OMR pipeline visualization."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.bubble_reader
import omr_utils.template_loader
import omr_utils.timing_mark_anchors


#============================================
def _compute_refinement_shift_data(results: list, template: dict,
	width: int, height: int, local_similarity_px: float = 2.5) -> dict:
	"""Compute template->refined shift vectors and local similarity flags.

	Returns:
		dict keyed by (question, choice) with:
		- template: (tx, ty)
		- refined: (rx, ry)
		- dx, dy
		- local_dev
		- local_ok
	"""
	choices = template["answers"]["choices"]
	left_range = template["answers"]["left_column"]["question_range"]
	right_range = template["answers"]["right_column"]["question_range"]
	shift_data = {}
	for entry in results:
		q_num = entry["question"]
		positions = entry.get("positions", {})
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			tx, ty = omr_utils.template_loader.to_pixels(
				norm_x, norm_y, width, height)
			rx, ry = positions.get(choice, (tx, ty))
			dx = rx - tx
			dy = ry - ty
			shift_data[(q_num, choice)] = {
				"template": (tx, ty),
				"refined": (rx, ry),
				"dx": dx,
				"dy": dy,
				"local_dev": 0.0,
				"local_ok": True,
			}
	for entry in results:
		q_num = entry["question"]
		if left_range[0] <= q_num <= left_range[1]:
			col_min = left_range[0]
			col_max = left_range[1]
		else:
			col_min = right_range[0]
			col_max = right_range[1]
		for choice in choices:
			key = (q_num, choice)
			current = shift_data[key]
			neighbors = []
			for delta in [-2, -1, 1, 2]:
				n_q = q_num + delta
				if n_q < col_min or n_q > col_max:
					continue
				n_key = (n_q, choice)
				if n_key in shift_data:
					neighbors.append(shift_data[n_key])
			if not neighbors:
				continue
			n_dx = [n["dx"] for n in neighbors]
			n_dy = [n["dy"] for n in neighbors]
			med_dx = float(numpy.median(n_dx))
			med_dy = float(numpy.median(n_dy))
			dev = ((current["dx"] - med_dx) ** 2 + (current["dy"] - med_dy) ** 2) ** 0.5
			current["local_dev"] = float(dev)
			current["local_ok"] = bool(dev <= local_similarity_px)
	return shift_data


#============================================
def _default_bounds(cx: int, cy: int, geom: dict) -> tuple:
	"""Compute integer pixel bounds from float geometry values.

	Args:
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		geom: pixel geometry dict (may contain float values)

	Returns:
		tuple of (top_y, bot_y, left_x, right_x) as integers
	"""
	top_y = int(cy - geom["half_height"])
	bot_y = int(cy + geom["half_height"])
	left_x = int(cx - geom["half_width"])
	right_x = int(cx + geom["half_width"])
	return (top_y, bot_y, left_x, right_x)


#============================================
def draw_answer_debug(image: numpy.ndarray, template: dict,
	results: list, show_refine_shifts: bool = True) -> numpy.ndarray:
	"""Draw color-coded rectangular bubble overlay on a registered image.

	Uses semi-transparent filled rectangles so every detection zone is
	visible even when zones share edges. Uses detected edge positions
	from read_answers results for accurate overlay alignment.

	- Teal filled strips = measurement zones (left and right of center)
	- Orange outline = center exclusion zone (printed letter area)
	- Status color outline = outer bubble border (green/red/yellow/gray)
	- Optional shift lines = template center to refined center; red when
	  local-neighbor shift similarity is poor

	Alpha blending (~0.3) keeps the underlying scantron image visible.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers
		show_refine_shifts: draw refinement shift vectors when True

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	# overlay layer for alpha-blended filled regions
	overlay = debug.copy()
	h, w = debug.shape[:2]
	choices = template["answers"]["choices"]
	# get geometry for fallback and center exclusion
	geom = omr_utils.bubble_reader.default_geom()
	shift_data = {}
	if show_refine_shifts:
		shift_data = _compute_refinement_shift_data(results, template, w, h)
	# int-cast geom values for cv2.rectangle drawing
	ce = int(geom["center_exclusion"])
	mi_v = int(geom["measurement_inset_v"])
	mi_h = int(geom["measurement_inset_h"])
	# define zone colors (BGR)
	teal = (200, 128, 0)
	orange = (0, 165, 255)
	alpha = 0.3
	for entry in results:
		q_num = entry["question"]
		positions = entry.get("positions", {})
		edges = entry.get("edges", {})
		for choice in choices:
			# use refined positions from read_answers if available
			if choice in positions:
				px, py = positions[choice]
			else:
				# fallback: recompute from template
				norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
					template, q_num, choice)
				px, py = omr_utils.template_loader.to_pixels(
					norm_x, norm_y, w, h)
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				# fallback to geometry-based defaults
				top_y, bot_y, left_x, right_x = _default_bounds(px, py, geom)
			# -- layer 1: teal filled measurement strips (alpha blended) --
			# matches _compute_edge_mean: inset from detected edges
			# left measurement strip
			cv2.rectangle(overlay,
				(left_x + mi_h, top_y + mi_v),
				(px - ce, bot_y - mi_v),
				teal, -1)
			# right measurement strip
			cv2.rectangle(overlay,
				(px + ce, top_y + mi_v),
				(right_x - mi_h, bot_y - mi_v),
				teal, -1)
	# blend the filled overlay onto the debug image
	cv2.addWeighted(overlay, alpha, debug, 1.0 - alpha, 0, debug)
	# draw outlines on top of the blended image (no alpha needed)
	for entry in results:
		q_num = entry["question"]
		answer = entry["answer"]
		flags = entry["flags"]
		scores = entry["scores"]
		positions = entry.get("positions", {})
		edges = entry.get("edges", {})
		for choice in choices:
			if choice in positions:
				px, py = positions[choice]
			else:
				norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
					template, q_num, choice)
				px, py = omr_utils.template_loader.to_pixels(
					norm_x, norm_y, w, h)
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				top_y, bot_y, left_x, right_x = _default_bounds(px, py, geom)
			# optional refinement-shift line for second-pass diagnostics
			if show_refine_shifts:
				shift_key = (q_num, choice)
				if shift_key in shift_data:
					sd = shift_data[shift_key]
					start_pt = sd["template"]
					end_pt = sd["refined"]
					dx = sd["dx"]
					dy = sd["dy"]
					if dx != 0 or dy != 0:
						if sd["local_ok"]:
							shift_color = (80, 200, 80)
						else:
							shift_color = (0, 0, 255)
						cv2.line(debug, start_pt, end_pt, shift_color, 1)
						cv2.circle(debug, start_pt, 1, shift_color, -1)
			# -- layer 3: orange center exclusion outline --
			cv2.rectangle(debug,
				(px - ce, top_y),
				(px + ce, bot_y),
				orange, 1)
			# -- layer 4: status-colored outer bubble outline --
			if choice == answer and "MULTIPLE" not in flags:
				# selected answer: green
				status_color = (0, 200, 0)
				thickness = 2
			elif choice == answer and "MULTIPLE" in flags:
				# selected but multiple: yellow
				status_color = (0, 255, 255)
				thickness = 2
			elif flags == "BLANK":
				# all blank: gray
				status_color = (128, 128, 128)
				thickness = 1
			else:
				# not selected: red
				status_color = (0, 0, 200)
				thickness = 1
			cv2.rectangle(debug,
				(left_x, top_y),
				(right_x, bot_y),
				status_color, thickness)
			# draw score text for high scores
			if scores[choice] > 0.10:
				score_text = f"{scores[choice]:.2f}"
				cv2.putText(debug, score_text,
					(px - 12, top_y - 2),
					cv2.FONT_HERSHEY_SIMPLEX, 0.25,
					status_color, 1)
	return debug


#============================================
def draw_scored_overlay(image: numpy.ndarray, template: dict,
	results: list) -> numpy.ndarray:
	"""Draw minimal scoring overlay showing bubble status and confidence.

	Shows filled/unfilled determination with confidence scores.
	No timing marks, no guide lines, no detection zones.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers

	Returns:
		annotated copy of the image
	"""
	# reuse existing answer debug without shift vectors for cleaner output
	scored = draw_answer_debug(image, template, results,
		show_refine_shifts=False)
	return scored


#============================================
def draw_combined_debug(image: numpy.ndarray, template: dict,
	transform: dict, results: list) -> numpy.ndarray:
	"""Draw combined debug overlay with all diagnostic layers.

	Layers (bottom to top):
	1. Timing mark candidates with cluster colors (search strips)
	2. Final timing marks with guide lines
	3. Bubble outlines with detection zones and shift vectors

	Args:
		image: BGR registered image
		template: loaded template dictionary
		transform: dict from estimate_anchor_transform
		results: list of answer dicts from read_answers

	Returns:
		annotated copy with all debug layers combined
	"""
	# start with timing candidates (search strips + cluster bboxes)
	debug = omr_utils.timing_mark_anchors.draw_timing_candidates_debug(
		image, transform)
	# overlay final timing marks and guide lines on top
	debug = omr_utils.timing_mark_anchors.draw_timing_mark_debug(
		debug, transform)
	# overlay answer bubbles with detection zones and shift vectors
	debug = draw_answer_debug(debug, template, results,
		show_refine_shifts=True)
	return debug
