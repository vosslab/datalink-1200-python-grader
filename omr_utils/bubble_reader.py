"""Read filled bubbles from a registered scantron image."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader

# Bubble box dimensions in pixels at canonical 1700x2200 resolution.
# The DataLink 1200 bubbles are rectangular bracket boxes (e.g. [A] [B])
# with a printed letter in the center. Scoring measures the left and
# right edges of the box where pencil fill appears, excluding the center
# where the printed letter always contributes darkness.
BUBBLE_HALF_WIDTH = 16
BUBBLE_HALF_HEIGHT = 10
# half-width of the center exclusion zone (printed letter area)
CENTER_EXCLUSION = 7


#============================================
def score_bubble_fast(gray: numpy.ndarray, cx: int, cy: int,
	radius: int) -> float:
	"""Score a single bubble by measuring its edge strip darkness.

	The DataLink 1200 form has rectangular bracket-shaped bubbles
	with a printed letter (A-E) in the center. This scorer measures
	the left and right edges of the bubble box where student pencil
	marks appear, excluding the center where the printed letter
	always contributes darkness regardless of fill state.

	Returns a self-referencing score: the lightest (emptiest) choice
	in the same row scores 0.0, filled choices score higher. This is
	done by read_answers using the raw edge means from this function.

	For standalone use (single bubble), returns the edge mean darkness
	relative to a local background reference.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		radius: bubble radius in pixels (used for bounds checking only)

	Returns:
		fill score (higher = more likely filled, range ~0.0 to 0.6)
		returns -1.0 for out-of-bounds coordinates
	"""
	h, w = gray.shape
	# bounds check
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return -1.0
	edge_mean = _compute_edge_mean(gray, cx, cy)
	if edge_mean < 0:
		return -1.0
	# for standalone use: background from strips above and below
	bg_gap = 2
	bg_height = 10
	bg_y1 = max(0, cy - BUBBLE_HALF_HEIGHT - bg_gap - bg_height)
	bg_y2 = max(0, cy - BUBBLE_HALF_HEIGHT - bg_gap)
	bg_y3 = min(h, cy + BUBBLE_HALF_HEIGHT + bg_gap)
	bg_y4 = min(h, cy + BUBBLE_HALF_HEIGHT + bg_gap + bg_height)
	lx1 = max(0, cx - BUBBLE_HALF_WIDTH)
	rx2 = min(w, cx + BUBBLE_HALF_WIDTH)
	bg_above = gray[bg_y1:bg_y2, lx1:rx2]
	bg_below = gray[bg_y3:bg_y4, lx1:rx2]
	bg_pixels = numpy.concatenate([bg_above.ravel(), bg_below.ravel()])
	if bg_pixels.size == 0:
		return 0.0
	bg_mean = float(numpy.mean(bg_pixels))
	score = (bg_mean - edge_mean) / 255.0
	return score


#============================================
def _compute_edge_mean(gray: numpy.ndarray, cx: int, cy: int) -> float:
	"""Compute mean brightness of the left and right edge strips.

	Measures the rectangular strips on either side of the center
	exclusion zone (where the printed letter sits). Lower values
	indicate darker (more filled) bubbles.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels
		cy: bubble center y in pixels

	Returns:
		average brightness of left+right edge strips (0-255 scale),
		or -1.0 if out of bounds
	"""
	h, w = gray.shape
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return -1.0
	# left edge strip: from left bracket to center exclusion boundary
	lx1 = max(0, cx - BUBBLE_HALF_WIDTH)
	lx2 = max(0, cx - CENTER_EXCLUSION)
	# right edge strip: from center exclusion boundary to right bracket
	rx1 = min(w, cx + CENTER_EXCLUSION)
	rx2 = min(w, cx + BUBBLE_HALF_WIDTH)
	# vertical bounds of the bubble box
	y1 = max(0, cy - BUBBLE_HALF_HEIGHT)
	y2 = min(h, cy + BUBBLE_HALF_HEIGHT)
	left_strip = gray[y1:y2, lx1:lx2]
	right_strip = gray[y1:y2, rx1:rx2]
	if left_strip.size == 0 or right_strip.size == 0:
		return -1.0
	# mean of both edge strips (lower = darker = more filled)
	edge_mean = (float(numpy.mean(left_strip))
		+ float(numpy.mean(right_strip))) / 2.0
	return edge_mean


#============================================
def _find_adaptive_threshold(spreads: list) -> float:
	"""Find the blank/filled threshold using the largest gap in sorted spreads.

	Each question produces a spread value (max edge_mean - min edge_mean
	across its 5 choices). Filled questions have large spreads (one choice
	is much darker), blank questions have small spreads (all similar).
	The natural gap between these two populations gives the threshold.

	Args:
		spreads: list of (question_number, spread_value) tuples

	Returns:
		adaptive threshold in pixels; questions above this are filled
	"""
	sorted_vals = sorted(s for _, s in spreads)
	max_gap = 0.0
	max_gap_idx = 0
	for i in range(len(sorted_vals) - 1):
		gap = sorted_vals[i + 1] - sorted_vals[i]
		if gap > max_gap:
			max_gap = gap
			max_gap_idx = i
	# threshold is the midpoint of the largest gap
	threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2.0
	return threshold


#============================================
def read_answers(image: numpy.ndarray, template: dict,
	multi_gap: float = 0.03) -> list:
	"""Read all 100 answers from a registered scantron image.

	Uses self-referencing scoring: for each question, the lightest
	(emptiest) choice in the row is used as the baseline. This avoids
	dependency on local background strips, which can be unreliable
	for phone photos with uneven lighting or machine-printed marks.

	Blank detection uses adaptive thresholding: the spread (max - min
	edge mean) across all 100 questions is analyzed to find the natural
	gap between filled and blank populations.

	Args:
		image: BGR registered image (perspective-corrected, canonical size)
		template: loaded template dictionary
		multi_gap: min spread gap between top two scores for MULTIPLE flag

	Returns:
		list of dicts with keys: question, answer, scores, flags
		where answer is a choice letter or empty string if blank,
		scores is a dict of choice->score, and flags is a string
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# light blur to reduce noise while preserving fill signal
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	h, w = image.shape[:2]
	answers_config = template["answers"]
	num_q = answers_config["num_questions"]
	choices = answers_config["choices"]
	# first pass: compute edge means for all questions and choices
	all_edge_means = []
	for q_num in range(1, num_q + 1):
		edge_means = {}
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			edge_means[choice] = _compute_edge_mean(gray, px, py)
		all_edge_means.append(edge_means)
	# compute per-question spread for adaptive blank detection
	spreads = []
	for q_idx, edge_means in enumerate(all_edge_means):
		vals = list(edge_means.values())
		spread = max(vals) - min(vals)
		spreads.append((q_idx + 1, spread))
	# find threshold that separates filled from blank questions
	blank_threshold = _find_adaptive_threshold(spreads)
	# second pass: build results using self-referencing scores
	results = []
	for q_idx, edge_means in enumerate(all_edge_means):
		q_num = q_idx + 1
		vals = list(edge_means.values())
		max_edge = max(vals)
		spread = max_edge - min(vals)
		# self-referencing score: how much darker than lightest choice
		scores = {}
		for choice in choices:
			scores[choice] = (max_edge - edge_means[choice]) / 255.0
		# sort choices by score descending
		ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
		best_choice = ranked[0][0]
		best_score = ranked[0][1]
		second_score = ranked[1][1]
		# gap between best and second best (in pixel units for multi check)
		gap_from_second = best_score - second_score
		flags = ""
		if spread < blank_threshold:
			# no bubble stands out enough: blank question
			answer = ""
			flags = "BLANK"
		elif gap_from_second < multi_gap:
			# two bubbles have similar high scores: likely multiple marks
			answer = best_choice
			other = ranked[1][0]
			flags = f"MULTIPLE({other})"
		else:
			answer = best_choice
		entry = {
			"question": q_num,
			"answer": answer,
			"scores": scores,
			"flags": flags,
		}
		results.append(entry)
	return results


#============================================
def draw_answer_debug(image: numpy.ndarray, template: dict,
	results: list) -> numpy.ndarray:
	"""Draw color-coded rectangular bubble overlay on a registered image.

	Green = selected answer, Red = empty bubble, Yellow = uncertain/multiple.
	Rectangles match the scoring region with center exclusion zone shown.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	h, w = debug.shape[:2]
	choices = template["answers"]["choices"]
	hw = BUBBLE_HALF_WIDTH
	hh = BUBBLE_HALF_HEIGHT
	for entry in results:
		q_num = entry["question"]
		answer = entry["answer"]
		flags = entry["flags"]
		scores = entry["scores"]
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			# color based on status
			if choice == answer and "MULTIPLE" not in flags:
				# selected answer: green
				color = (0, 200, 0)
				thickness = 2
			elif choice == answer and "MULTIPLE" in flags:
				# selected but multiple: yellow
				color = (0, 255, 255)
				thickness = 2
			elif flags == "BLANK":
				# all blank: gray
				color = (128, 128, 128)
				thickness = 1
			else:
				# not selected: red
				color = (0, 0, 200)
				thickness = 1
			# draw rectangle matching scoring region
			pt1 = (px - hw, py - hh)
			pt2 = (px + hw, py + hh)
			cv2.rectangle(debug, pt1, pt2, color, thickness)
			# draw score text for high scores
			if scores[choice] > 0.10:
				score_text = f"{scores[choice]:.2f}"
				cv2.putText(debug, score_text, (px - 12, py - hh - 2),
					cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
	return debug
