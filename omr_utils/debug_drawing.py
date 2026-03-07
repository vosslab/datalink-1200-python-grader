"""Debug overlay drawing for OMR pipeline visualization."""

# Standard Library
import math

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.slot_map
import omr_utils.timing_mark_anchors


#============================================
def draw_answer_debug(image: numpy.ndarray, template: dict,
	results: list, measure_cfg: dict,
	slot_map: "omr_utils.slot_map.SlotMap") -> numpy.ndarray:
	"""Draw color-coded rectangular bubble overlay on a registered image.

	Uses semi-transparent filled rectangles so every detection zone is
	visible even when zones share edges. Uses detected edge positions
	from read_answers results for accurate overlay alignment.

	- Teal filled strips = measurement zones (left and right of center)
	- Orange outline = center exclusion zone (printed letter area)
	- Status color outline = outer bubble border (green/red/yellow/gray)

	Alpha blending (~0.3) keeps the underlying scantron image visible.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		slot_map: SlotMap instance for lattice-based ROI bounds

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	# overlay layer for alpha-blended filled regions
	overlay = debug.copy()
	h, w = debug.shape[:2]
	choices = template["answers"]["choices"]
	# use provided geometry for center exclusion and measurement insets
	ce = measure_cfg["center_exclusion"]
	mi_v = measure_cfg["measurement_inset_v"]
	mi_h = measure_cfg["measurement_inset_h"]
	# define zone colors (BGR)
	teal = (200, 128, 0)
	orange = (0, 165, 255)
	alpha = 0.3
	for entry in results:
		q_num = entry["question"]
		positions = entry.get("positions", {})
		edges = entry.get("edges", {})
		for choice in choices:
			# use positions from read_answers results
			if choice not in positions:
				continue
			px, py = positions[choice]
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				# fallback to lattice-based ROI bounds
				top_y, bot_y, left_x, right_x = slot_map.roi_bounds(
					q_num, choice)
			# -- layer 1: teal filled measurement strips (alpha blended) --
			# matches _compute_edge_mean: inset from detected edges
			# left measurement strip
			cv2.rectangle(overlay,
				(int(left_x + mi_h), int(top_y + mi_v)),
				(int(px - ce), int(bot_y - mi_v)),
				teal, -1)
			# right measurement strip
			cv2.rectangle(overlay,
				(int(px + ce), int(top_y + mi_v)),
				(int(right_x - mi_h), int(bot_y - mi_v)),
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
			if choice not in positions:
				continue
			px, py = positions[choice]
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				# fallback to lattice-based ROI bounds
				top_y, bot_y, left_x, right_x = slot_map.roi_bounds(
					q_num, choice)
			# -- layer 3: orange center exclusion outline --
			cv2.rectangle(debug,
				(int(px - ce), int(top_y)),
				(int(px + ce), int(bot_y)),
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
				(int(left_x), int(top_y)),
				(int(right_x), int(bot_y)),
				status_color, thickness)
			# draw score text for high scores
			if scores[choice] > 0.10:
				score_text = f"{scores[choice]:.2f}"
				cv2.putText(debug, score_text,
					(int(px) - 12, int(top_y) - 2),
					cv2.FONT_HERSHEY_SIMPLEX, 0.25,
					status_color, 1)
			# template refinement confidence tier indicator
			ref_conf = entry.get("refinement_confidence", {})
			if isinstance(ref_conf, dict):
				conf_val = ref_conf.get(choice, -1.0)
			else:
				conf_val = -1.0
			if conf_val >= 0.0:
				# pick tier color
				if conf_val >= 0.6:
					tier_color = (0, 200, 0)
				elif conf_val >= 0.3:
					tier_color = (0, 200, 200)
				else:
					tier_color = (0, 0, 200)
				# small dot at bottom-right corner of bubble
				cv2.circle(debug, (int(right_x) - 2, int(bot_y) - 2),
					2, tier_color, -1)
	return debug


#============================================
def draw_scored_overlay(image: numpy.ndarray, template: dict,
	results: list, measure_cfg: dict,
	slot_map: "omr_utils.slot_map.SlotMap") -> numpy.ndarray:
	"""Draw minimal scoring overlay showing bubble status and confidence.

	Shows filled/unfilled determination with confidence scores.
	No timing marks, no guide lines, no detection zones.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		slot_map: SlotMap instance for lattice-based ROI bounds

	Returns:
		annotated copy of the image
	"""
	# reuse existing answer debug for cleaner output
	scored = draw_answer_debug(image, template, results, measure_cfg,
		slot_map=slot_map)
	return scored


#============================================
def draw_lattice_crosshairs(image: numpy.ndarray,
	slot_map: "omr_utils.slot_map.SlotMap",
	template: dict) -> numpy.ndarray:
	"""Draw small crosshairs at every SlotMap.center() position.

	No ROIs, no measurement zones. Just lattice intersections.
	Verifies geometry independently of measurement logic. If crosses
	land on printed bubble centers, the geometry layer is correct.

	Args:
		image: BGR image
		slot_map: SlotMap instance
		template: loaded template dictionary

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	answers = template["answers"]
	num_q = answers["num_questions"]
	choices = answers["choices"]
	green = (0, 220, 0)
	cyan = (220, 220, 0)
	# crosshair arm length in pixels
	arm = 4
	for q_num in range(1, num_q + 1):
		for choice in choices:
			cx, cy = slot_map.center(q_num, choice)
			# draw small crosshair
			cv2.line(debug, (cx - arm, cy), (cx + arm, cy), green, 1)
			cv2.line(debug, (cx, cy - arm), (cx, cy + arm), green, 1)
		# label first row of each column for orientation
		if q_num == 1:
			cx, cy = slot_map.center(1, "A")
			cv2.putText(debug, "Q1", (cx - 10, cy - 8),
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, cyan, 1)
		elif q_num == 51:
			cx, cy = slot_map.center(51, "A")
			cv2.putText(debug, "Q51", (cx - 12, cy - 8),
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, cyan, 1)
	return debug


# column role labels for the 15-column local lattice
_COLUMN_LABELS = {
	0: "Q#L",
	1: "A", 2: "B", 3: "C", 4: "D", 5: "E",
	6: "gap",
	7: "Q#R",
	8: "A", 9: "B", 10: "C", 11: "D", 12: "E",
	13: "mrg", 14: "mrg",
}


#============================================
def draw_column_lattice(image: numpy.ndarray,
	transform: dict, num_columns: int = 15) -> numpy.ndarray:
	"""Draw vertical guide lines for logical columns 0 through num_columns-1.

	Each line is drawn at x_i = fp_x0 + i * col_pitch and labeled with
	its column index and role (Q#, A-E, gap, margin).

	Args:
		image: BGR image to annotate
		transform: dict from estimate_anchor_transform containing
			top_fp_x0 and top_col_spacing
		num_columns: number of logical columns to draw (default 15)

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	h, w = debug.shape[:2]
	fp_x0 = float(transform.get("top_fp_x0", 0.0))
	col_pitch = float(transform.get("top_col_spacing", 0.0))
	if fp_x0 <= 0 or col_pitch <= 0 or not math.isfinite(col_pitch):
		return debug
	# color scheme: answer cols cyan, Q# cols yellow, gap/margin gray
	color_answer = (220, 220, 0)
	color_qnum = (0, 220, 220)
	color_gap = (128, 128, 128)
	for i in range(num_columns):
		x = int(round(fp_x0 + i * col_pitch))
		if x < 0 or x >= w:
			continue
		label = _COLUMN_LABELS.get(i, str(i))
		# pick color based on role
		if label in ("A", "B", "C", "D", "E"):
			color = color_answer
		elif label.startswith("Q#"):
			color = color_qnum
		else:
			color = color_gap
		# draw vertical guide line
		cv2.line(debug, (x, 0), (x, h), color, 1)
		# draw label near top of image
		cv2.putText(debug, f"{i}:{label}", (x + 2, 14),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
	return debug


#============================================
def draw_combined_debug(image: numpy.ndarray, template: dict,
	transform: dict, results: list, measure_cfg: dict,
	slot_map: "omr_utils.slot_map.SlotMap") -> numpy.ndarray:
	"""Draw combined debug overlay with all diagnostic layers.

	Layers (bottom to top):
	1. Timing mark candidates with cluster colors (search strips)
	2. Final timing marks with guide lines
	3. Bubble outlines with detection zones

	Args:
		image: BGR registered image
		template: loaded template dictionary
		transform: dict from estimate_anchor_transform
		results: list of answer dicts from read_answers
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		slot_map: SlotMap instance for lattice-based ROI bounds

	Returns:
		annotated copy with all debug layers combined
	"""
	# start with timing candidates (search strips + cluster bboxes)
	debug = omr_utils.timing_mark_anchors.draw_timing_candidates_debug(
		image, transform)
	# overlay final timing marks and guide lines on top
	debug = omr_utils.timing_mark_anchors.draw_timing_mark_debug(
		debug, transform)
	# overlay answer bubbles with detection zones
	debug = draw_answer_debug(debug, template, results, measure_cfg,
		slot_map=slot_map)
	return debug
