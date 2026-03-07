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
	"""Draw three-zone color-coded bubble overlay on a registered image.

	Uses semi-transparent filled rectangles so every detection zone is
	visible even when zones share edges. Uses detected edge positions
	from read_answers results for accurate overlay alignment.

	Three zones per target.png:
	- Teal filled = green fill measurement windows (small interior rects)
	- Red filled = bracket bar reference strips (four narrow strips)
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
	# three-zone geometry from measure_cfg
	ce = measure_cfg["center_exclusion"]
	fi_v = measure_cfg["fill_inset_v"]
	bi = measure_cfg["bracket_inner_half"]
	bb_v = measure_cfg["bracket_bar_v"]
	bb_h = measure_cfg["bracket_bar_h"]
	# define zone colors (BGR)
	teal = (200, 128, 0)
	red = (0, 0, 200)
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
			# -- layer 1: teal filled fill-zone windows (alpha blended) --
			# green fill zones: small interior windows between bracket
			# bars and center letter, vertically between horizontal bars
			# left fill window
			cv2.rectangle(overlay,
				(int(px - bi), int(top_y + fi_v)),
				(int(px - ce), int(bot_y - fi_v)),
				teal, -1)
			# right fill window
			cv2.rectangle(overlay,
				(int(px + ce), int(top_y + fi_v)),
				(int(px + bi), int(bot_y - fi_v)),
				teal, -1)
			# -- layer 2: red bracket bar reference strips (alpha blended) --
			# four narrow strips on bracket horizontal bars (L-R, U-D mirrored)
			# top-left bracket bar strip
			cv2.rectangle(overlay,
				(int(px - bi), int(top_y + bb_v)),
				(int(px - ce), int(top_y + bb_v + bb_h)),
				red, -1)
			# top-right bracket bar strip
			cv2.rectangle(overlay,
				(int(px + ce), int(top_y + bb_v)),
				(int(px + bi), int(top_y + bb_v + bb_h)),
				red, -1)
			# bottom-left bracket bar strip
			cv2.rectangle(overlay,
				(int(px - bi), int(bot_y - bb_v - bb_h)),
				(int(px - ce), int(bot_y - bb_v)),
				red, -1)
			# bottom-right bracket bar strip
			cv2.rectangle(overlay,
				(int(px + ce), int(bot_y - bb_v - bb_h)),
				(int(px + bi), int(bot_y - bb_v)),
				red, -1)
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
def draw_student_id_overlay(image: numpy.ndarray, template: dict,
	slot_map: "omr_utils.slot_map.SlotMap") -> numpy.ndarray:
	"""Draw student ID bubble ROI rectangles and center crosshairs.

	Shows all 9x10 student ID positions with:
	- Cyan ROI rectangles from sid_roi_bounds()
	- Green center crosshairs from sid_center()
	- Digit column labels (D0-D8) and value row labels (0-9)

	Args:
		image: BGR registered image
		template: loaded template dictionary
		slot_map: SlotMap instance with student ID geometry

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	sid_config = template.get("student_id", {})
	num_digits = sid_config.get("num_digits", 9)
	num_values = sid_config.get("num_values", 10)
	# colors
	cyan = (220, 220, 0)
	green = (0, 220, 0)
	yellow = (0, 220, 220)
	# crosshair arm length
	arm = 3
	for d in range(num_digits):
		for v in range(num_values):
			# draw ROI rectangle
			top_y, bot_y, left_x, right_x = slot_map.sid_roi_bounds(d, v)
			cv2.rectangle(debug,
				(int(left_x), int(top_y)),
				(int(right_x), int(bot_y)),
				cyan, 1)
			# draw center crosshair
			cx, cy = slot_map.sid_center(d, v)
			cv2.line(debug, (cx - arm, cy), (cx + arm, cy), green, 1)
			cv2.line(debug, (cx, cy - arm), (cx, cy + arm), green, 1)
		# label digit column at top
		cx, cy = slot_map.sid_center(d, 0)
		label = f"D{d}"
		cv2.putText(debug, label, (cx - 6, cy - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, yellow, 1)
	# label value rows on the left side
	for v in range(num_values):
		cx, cy = slot_map.sid_center(0, v)
		label = str(v)
		cv2.putText(debug, label, (cx - 18, cy + 3),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, yellow, 1)
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
	# overlay student ID positions
	debug = draw_student_id_overlay(debug, template, slot_map)
	return debug


#============================================
def draw_ncc_shift_overlay(image: numpy.ndarray,
	results: list) -> numpy.ndarray:
	"""Draw triple-dot overlay showing NCC seed, peak, and final positions.

	Three dots per bubble:
	- Yellow (3px): lattice seed center (before NCC)
	- Magenta (3px): NCC peak center (after NCC, before shift filter)
	- Cyan (3px): final applied center (after shift filter)
	- Thin white line from seed to NCC peak (shows shift vector)

	Args:
		image: BGR registered image
		results: list of answer dicts from read_answers with
			ncc_positions data propagated

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	# colors (BGR)
	yellow = (0, 255, 255)
	magenta = (255, 0, 255)
	cyan = (255, 255, 0)
	dark_gray = (80, 80, 80)
	# plus-sign arm length in pixels
	arm = 3
	alpha = 0.5
	# draw each marker layer on a separate overlay for transparency
	# layer 1: white shift vector lines
	overlay_lines = debug.copy()
	for entry in results:
		ncc_pos = entry.get("ncc_positions", {})
		for choice, ncc_data in ncc_pos.items():
			seed_cx = int(round(ncc_data["seed_cx"]))
			seed_cy = int(round(ncc_data["seed_cy"]))
			ncc_cx = int(round(ncc_data["ncc_cx"]))
			ncc_cy = int(round(ncc_data["ncc_cy"]))
			cv2.line(overlay_lines, (seed_cx, seed_cy),
				(ncc_cx, ncc_cy), dark_gray, 1)
	cv2.addWeighted(overlay_lines, alpha, debug, 1.0 - alpha, 0, debug)
	# layer 2: yellow seed plus markers
	overlay_seed = debug.copy()
	for entry in results:
		ncc_pos = entry.get("ncc_positions", {})
		for choice, ncc_data in ncc_pos.items():
			cx = int(round(ncc_data["seed_cx"]))
			cy = int(round(ncc_data["seed_cy"]))
			cv2.line(overlay_seed, (cx - arm, cy),
				(cx + arm, cy), yellow, 1)
			cv2.line(overlay_seed, (cx, cy - arm),
				(cx, cy + arm), yellow, 1)
	cv2.addWeighted(overlay_seed, alpha, debug, 1.0 - alpha, 0, debug)
	# layer 3: magenta NCC peak plus markers
	overlay_ncc = debug.copy()
	for entry in results:
		ncc_pos = entry.get("ncc_positions", {})
		for choice, ncc_data in ncc_pos.items():
			cx = int(round(ncc_data["ncc_cx"]))
			cy = int(round(ncc_data["ncc_cy"]))
			cv2.line(overlay_ncc, (cx - arm, cy),
				(cx + arm, cy), magenta, 1)
			cv2.line(overlay_ncc, (cx, cy - arm),
				(cx, cy + arm), magenta, 1)
	cv2.addWeighted(overlay_ncc, alpha, debug, 1.0 - alpha, 0, debug)
	# layer 4: cyan final applied plus markers
	overlay_final = debug.copy()
	for entry in results:
		ncc_pos = entry.get("ncc_positions", {})
		positions = entry.get("positions", {})
		for choice, ncc_data in ncc_pos.items():
			if choice in positions:
				fx, fy = positions[choice]
				cx = int(round(fx))
				cy = int(round(fy))
			else:
				cx = int(round(ncc_data["seed_cx"]))
				cy = int(round(ncc_data["seed_cy"]))
			cv2.line(overlay_final, (cx - arm, cy),
				(cx + arm, cy), cyan, 1)
			cv2.line(overlay_final, (cx, cy - arm),
				(cx, cy + arm), cyan, 1)
	cv2.addWeighted(overlay_final, alpha, debug, 1.0 - alpha, 0, debug)
	return debug
