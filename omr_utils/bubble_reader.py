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
# vertical gap between bubble box edge and background sample strip
BG_GAP = 2
# height of background sample strip above and below the box
BG_HEIGHT = 10


#============================================
def score_bubble_fast(gray: numpy.ndarray, cx: int, cy: int,
	radius: int) -> float:
	"""Score a bubble using rectangular left+right edge strips.

	The DataLink 1200 form has rectangular bracket-shaped bubbles
	with a printed letter (A-E) in the center. This scorer measures
	the left and right edges of the bubble box where student pencil
	marks appear, excluding the center where the printed letter
	always contributes darkness regardless of fill state.

	The background reference is sampled from strips above and below
	the bubble box (between rows), which should be clean paper.

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
	# left edge strip: from left bracket to center exclusion boundary
	lx1 = max(0, cx - BUBBLE_HALF_WIDTH)
	lx2 = max(0, cx - CENTER_EXCLUSION)
	# right edge strip: from center exclusion boundary to right bracket
	rx1 = min(w, cx + CENTER_EXCLUSION)
	rx2 = min(w, cx + BUBBLE_HALF_WIDTH)
	# vertical bounds of the bubble box
	y1 = max(0, cy - BUBBLE_HALF_HEIGHT)
	y2 = min(h, cy + BUBBLE_HALF_HEIGHT)
	# extract left and right edge strips
	left_strip = gray[y1:y2, lx1:lx2]
	right_strip = gray[y1:y2, rx1:rx2]
	if left_strip.size == 0 or right_strip.size == 0:
		return -1.0
	# mean darkness of edge strips (lower = darker = more filled)
	edge_mean = (float(numpy.mean(left_strip))
		+ float(numpy.mean(right_strip))) / 2.0
	# background reference: paper strips above and below the box
	bg_y1 = max(0, cy - BUBBLE_HALF_HEIGHT - BG_GAP - BG_HEIGHT)
	bg_y2 = max(0, cy - BUBBLE_HALF_HEIGHT - BG_GAP)
	bg_y3 = min(h, cy + BUBBLE_HALF_HEIGHT + BG_GAP)
	bg_y4 = min(h, cy + BUBBLE_HALF_HEIGHT + BG_GAP + BG_HEIGHT)
	bg_above = gray[bg_y1:bg_y2, lx1:rx2]
	bg_below = gray[bg_y3:bg_y4, lx1:rx2]
	bg_pixels = numpy.concatenate([bg_above.ravel(), bg_below.ravel()])
	if bg_pixels.size == 0:
		return 0.0
	bg_mean = float(numpy.mean(bg_pixels))
	# score: how much darker the edges are compared to paper background
	score = (bg_mean - edge_mean) / 255.0
	return score


#============================================
def read_answers(image: numpy.ndarray, template: dict,
	blank_gap: float = 0.10, multi_gap: float = 0.08) -> list:
	"""Read all 100 answers from a registered scantron image.

	Uses per-question relative scoring: the choice with the highest
	score is selected. A question is BLANK if the best choice does
	not stand out enough from the mean of all choices. MULTIPLE is
	flagged if two choices have similar high scores.

	Args:
		image: BGR registered image (perspective-corrected, canonical size)
		template: loaded template dictionary
		blank_gap: min gap between max score and mean to be non-blank
		multi_gap: min gap between top two scores to avoid MULTIPLE flag

	Returns:
		list of dicts with keys: question, answer, scores, flags
		where answer is a choice letter or empty string if blank,
		scores is a dict of choice->score, and flags is a string
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# light blur to reduce noise while preserving fill signal
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	h, w = image.shape[:2]
	radius = omr_utils.template_loader.get_bubble_radius_px(template, w, h)
	answers_config = template["answers"]
	num_q = answers_config["num_questions"]
	choices = answers_config["choices"]
	results = []
	for q_num in range(1, num_q + 1):
		scores = {}
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			score = score_bubble_fast(gray, px, py, radius)
			scores[choice] = score
		# sort choices by score descending
		ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
		best_choice = ranked[0][0]
		best_score = ranked[0][1]
		second_score = ranked[1][1]
		# mean score across all choices
		mean_score = sum(s for _, s in ranked) / len(ranked)
		# gap between best and mean
		gap_from_mean = best_score - mean_score
		# gap between best and second best
		gap_from_second = best_score - second_score
		flags = ""
		if gap_from_mean < blank_gap:
			# no bubble stands out: blank question
			answer = ""
			flags = "BLANK"
		elif gap_from_second < multi_gap:
			# two bubbles are close: likely multiple marks
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
