"""SlotMap: single source of truth for bubble slot geometry.

Builds pixel coordinates directly from timing mark anchors.
Row centers come from left question marks, column centers from
top footprint origin + choice_columns * local lattice column pitch.
No YAML coordinates, no affine correction, no Sobel refinement.
"""

# Standard Library
import math


# Detection zone ratios: dimensionless fractions relating bubble
# measurement geometry to timing-mark spacing. Horizontal ratios
# scale with col_pitch, vertical ratios with row_pitch.
# Three-zone model per target.png:
#   Orange = center exclusion (letter glyph, avoid entirely)
#   Red    = bracket bar reference strips (narrow bars at top/bottom)
#   Green  = fill measurement windows (between bars and center letter)
# Calibrated from artifacts/base_letter_template.png (480x88) using
# tools/calibrate_bubble_geometry.py. Measurements are L-R and U-D
# symmetrized. Calibration date: 2026-03-10.
_DZ_CENTER_EXCLUSION = 0.0896  # center letter glyph half-width / col_pitch
_DZ_BRACKET_INNER_HALF = 0.3104  # half-width from cx to bracket inner face / col_pitch
_DZ_FILL_INSET_V = 0.3864     # fill zone top (below bracket bar) / row_pitch
_DZ_BRACKET_BAR_V = 0.3295    # bracket bar top edge / row_pitch
_DZ_BRACKET_BAR_H = 0.0455    # bracket bar thickness / row_pitch
_DZ_REFINE_MAX_SHIFT = 0.3210 # max template shift / row_pitch
_DZ_REFINE_PAD_V = 0.1710     # vertical refine padding / row_pitch
_DZ_REFINE_PAD_H = 0.0833     # horizontal refine padding / col_pitch


#============================================
class SlotMap:
	"""Single geometry authority for bubble slot positions.

	Built directly from estimate_anchor_transform() output and
	template config. All pixel coordinates derive from the
	timing-mark lattice: row y from left marks, column x from
	top footprint origin + column index * local lattice column pitch.

	Args:
		transform: dict from estimate_anchor_transform() containing
			top_fp_x0, top_col_spacing, left_question_marks, left_s_q
		template: dict containing choice_columns, question_range, choices
	"""

	#============================================
	def __init__(self, transform: dict, template: dict):
		"""Initialize SlotMap from timing mark transform and template."""
		# extract timing-mark lattice data
		fp_x0 = float(transform.get("top_fp_x0", 0.0))
		# local lattice column pitch (pixels per column)
		col_pitch = float(transform.get("top_col_spacing", 0.0))
		if fp_x0 <= 0 or col_pitch <= 0:
			raise ValueError(
				f"SlotMap: top footprint unavailable "
				f"(fp_x0={fp_x0}, col_pitch={col_pitch})")
		question_marks = transform.get("left_question_marks", [])
		if len(question_marks) < 50:
			raise ValueError(
				f"SlotMap: need 50 question marks, "
				f"got {len(question_marks)}")
		# store pitches
		self._row_pitch = float(transform.get("left_s_q", 0.0))
		self._col_pitch = float(col_pitch)
		if (not math.isfinite(self._row_pitch)
			or not math.isfinite(self._col_pitch)
			or self._row_pitch <= 0 or self._col_pitch <= 0):
			raise ValueError(
				f"SlotMap: invalid spacing "
				f"(row_pitch={self._row_pitch}, col_pitch={self._col_pitch})")
		# extract template config
		answers = template["answers"]
		self._choices = list(answers["choices"])
		self._num_questions = int(answers["num_questions"])
		left_col = answers["left_column"]
		right_col = answers["right_column"]
		self._left_range = tuple(left_col["question_range"])
		self._right_range = tuple(right_col["question_range"])
		left_cols = left_col.get("choice_columns", {})
		right_cols = right_col.get("choice_columns", {})
		# build row y-centers from left question marks
		self._row_y = []
		for i in range(50):
			cy = float(question_marks[i]["center_y"])
			self._row_y.append(cy)
		# build column x-centers for each (side, choice)
		self._col_x = {}
		for choice in self._choices:
			if choice not in left_cols:
				raise ValueError(
					f"SlotMap: choice '{choice}' missing from "
					f"left choice_columns")
			if choice not in right_cols:
				raise ValueError(
					f"SlotMap: choice '{choice}' missing from "
					f"right choice_columns")
			self._col_x[("left", choice)] = fp_x0 + left_cols[choice] * col_pitch
			self._col_x[("right", choice)] = fp_x0 + right_cols[choice] * col_pitch
		# --- student ID geometry from timing marks ---
		id_marks = transform.get("left_id_marks", [])
		sid_config = template.get("student_id", {})
		self._sid_col_indices = sid_config.get("id_columns", [])
		self._sid_row_pitch = float(transform.get("left_s_id", 0.0))
		# build student ID row y-centers from left_id_marks
		self._sid_row_y = []
		for mark in id_marks:
			self._sid_row_y.append(float(mark["center_y"]))
		# build student ID column x-centers from lattice
		self._sid_col_x = []
		for col_idx in self._sid_col_indices:
			x = fp_x0 + col_idx * col_pitch
			self._sid_col_x.append(x)

	#============================================
	@property
	def row_pitch(self) -> float:
		"""Row spacing in pixels."""
		return self._row_pitch

	#============================================
	@property
	def col_pitch(self) -> float:
		"""Column spacing in pixels."""
		return self._col_pitch

	#============================================
	def _side_and_row_idx(self, q_num: int) -> tuple:
		"""Return (side_str, row_index) for a question number.

		Args:
			q_num: question number (1-based)

		Returns:
			tuple of ('left' or 'right', row_index)
		"""
		if self._left_range[0] <= q_num <= self._left_range[1]:
			row_idx = q_num - self._left_range[0]
			return ("left", row_idx)
		if self._right_range[0] <= q_num <= self._right_range[1]:
			row_idx = q_num - self._right_range[0]
			return ("right", row_idx)
		raise ValueError(f"SlotMap: question {q_num} not in any column range")

	#============================================
	def row_center(self, q_num: int) -> int:
		"""Return cy pixel coord for a question row.

		Args:
			q_num: question number (1-based)

		Returns:
			integer y pixel coordinate
		"""
		_, row_idx = self._side_and_row_idx(q_num)
		cy = int(round(self._row_y[row_idx]))
		return cy

	#============================================
	def choice_center(self, q_num: int, choice: str) -> int:
		"""Return cx pixel coord for a choice in the correct column side.

		Args:
			q_num: question number (1-based)
			choice: choice letter (A-E)

		Returns:
			integer x pixel coordinate
		"""
		side, _ = self._side_and_row_idx(q_num)
		cx = int(round(self._col_x[(side, choice)]))
		return cx

	#============================================
	def center(self, q_num: int, choice: str) -> tuple:
		"""Return (cx, cy) pixel coords for question/choice slot.

		Args:
			q_num: question number (1-based)
			choice: choice letter (A-E)

		Returns:
			tuple of (cx, cy) integer pixel coords
		"""
		cx = self.choice_center(q_num, choice)
		cy = self.row_center(q_num)
		return (cx, cy)

	#============================================
	def roi_bounds(self, q_num: int, choice: str) -> tuple:
		"""Return (top_y, bot_y, left_x, right_x) from lattice midpoints.

		Interior slots use midpoints between adjacent centers.
		Edge slots (A/E, first/last row) extrapolate by half pitch.

		Args:
			q_num: question number (1-based)
			choice: choice letter (A-E)

		Returns:
			tuple of (top_y, bot_y, left_x, right_x) as integers
		"""
		cx, cy = self.center(q_num, choice)
		side, row_idx = self._side_and_row_idx(q_num)
		choice_idx = self._choices.index(choice)
		# --- horizontal bounds from neighboring choice centers ---
		x_vals = []
		for c in self._choices:
			x_vals.append(self._col_x[(side, c)])
		i = choice_idx
		if 0 < i < len(self._choices) - 1:
			# interior choice: midpoint to neighbors
			left_x = int(round((x_vals[i - 1] + x_vals[i]) / 2.0))
			right_x = int(round((x_vals[i] + x_vals[i + 1]) / 2.0))
		elif i == 0:
			# first choice (A): extrapolate left by half pitch
			left_x = int(round(x_vals[0] - self._col_pitch / 2.0))
			right_x = int(round((x_vals[0] + x_vals[1]) / 2.0))
		else:
			# last choice (E): extrapolate right by half pitch
			left_x = int(round((x_vals[-2] + x_vals[-1]) / 2.0))
			right_x = int(round(x_vals[-1] + self._col_pitch / 2.0))
		# --- vertical bounds from neighboring row centers ---
		if 0 < row_idx < 49:
			# interior row: midpoint to neighbors
			top_y = int(round(
				(self._row_y[row_idx - 1] + self._row_y[row_idx]) / 2.0))
			bot_y = int(round(
				(self._row_y[row_idx] + self._row_y[row_idx + 1]) / 2.0))
		elif row_idx == 0:
			# first row: extrapolate top by half pitch
			top_y = int(round(
				self._row_y[0] - self._row_pitch / 2.0))
			bot_y = int(round(
				(self._row_y[0] + self._row_y[1]) / 2.0))
		else:
			# last row: extrapolate bottom by half pitch
			top_y = int(round(
				(self._row_y[48] + self._row_y[49]) / 2.0))
			bot_y = int(round(
				self._row_y[49] + self._row_pitch / 2.0))
		return (top_y, bot_y, left_x, right_x)

	#============================================
	def print_lattice_diagnostic(self) -> None:
		"""Print column lattice positions, pitches, and answer mapping.

		Diagnostic output for verifying that col_pitch and row_pitch
		produce correct bubble positions on the printed form. Judges
		correctness by whether lattice lines land on printed structures.
		"""
		# column role labels for the 15-column local lattice
		col_labels = {
			0: "Q#L", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E",
			6: "gap", 7: "Q#R", 8: "A", 9: "B", 10: "C", 11: "D", 12: "E",
			13: "mrg", 14: "mrg",
		}
		# print column lattice positions
		print("Column lattice (x = x0 + col * col_pitch):")
		print(f"  {'col':>3}  {'x_px':>6}  role")
		# compute fp_x0 from column 0 center: x0 = col_x[(left,A)] - 1*col_pitch
		# since left A is at column index 1
		fp_x0 = self._col_x[("left", "A")] - 1.0 * self._col_pitch
		for col_idx in range(15):
			x_px = fp_x0 + col_idx * self._col_pitch
			label = col_labels.get(col_idx, str(col_idx))
			print(f"  {col_idx:3d}  {x_px:6.1f}  {label}")
		# print pitches and ratio
		ratio = self._col_pitch / self._row_pitch if self._row_pitch > 0 else 0.0
		print(f"\nPitches: col_pitch={self._col_pitch:.1f}px"
			f"  row_pitch={self._row_pitch:.1f}px"
			f"  ratio={ratio:.3f}")
		# print concrete A-E positions for left and right sides
		print("\nAnswer mapping:")
		for side_label, side in [("left", "left"), ("right", "right")]:
			parts = []
			for choice in self._choices:
				cx = self._col_x[(side, choice)]
				parts.append(f"{choice}={cx:.1f}")
			print(f"  {side_label}: {', '.join(parts)}")

	#============================================
	def print_roi_diagnostic(self) -> None:
		"""Print ROI width/height/aspect for representative slots.

		Samples Q10, Q50 with choices A, C, E to verify that ROI
		dimensions are landscape (width > height) after col_pitch fix.
		"""
		print("\nROI diagnostics (width x height, aspect=w/h):")
		sample_slots = [
			(10, "A"), (10, "C"), (10, "E"),
			(50, "A"), (50, "C"), (50, "E"),
		]
		for q_num, choice in sample_slots:
			top_y, bot_y, left_x, right_x = self.roi_bounds(
				q_num, choice)
			w = right_x - left_x
			h = bot_y - top_y
			# avoid divide-by-zero
			aspect = w / h if h > 0 else 0.0
			print(f"  Q{q_num:2d}-{choice}:  "
				f"left={left_x} right={right_x} "
				f"top={top_y} bot={bot_y}  "
				f"{w}x{h}  aspect={aspect:.2f}")

	#============================================
	def print_seam_diagnostic(self) -> None:
		"""Print seam errors between adjacent choice ROIs.

		For a representative row (Q10), prints the gap or overlap
		between right edge of one choice and left edge of the next.
		Expected seam error is at most 1 pixel (rounding only).
		"""
		q_num = 10
		side, _ = self._side_and_row_idx(q_num)
		print(f"\nSeam diagnostics (Q{q_num}, {side} side):")
		for i in range(len(self._choices) - 1):
			# get right edge of current choice
			c_left = self._choices[i]
			c_right = self._choices[i + 1]
			_, _, _, right_edge = self.roi_bounds(q_num, c_left)
			_, _, left_edge, _ = self.roi_bounds(q_num, c_right)
			# seam error: should be 0 (or at most 1 from rounding)
			seam_error = left_edge - right_edge
			print(f"  {c_left}->{c_right}: seam_error = {seam_error} px")

	#============================================
	def measure_cfg(self) -> dict:
		"""Return measurement constants derived from timing-mark spacing.

		Three-zone model per target.png:
		  fill zones = green interior windows for fill measurement
		  bracket bars = red reference strips on horizontal bars
		  center exclusion = orange letter glyph zone

		Returns:
			dict with keys: center_exclusion, bracket_inner_half,
			fill_inset_v, bracket_bar_v, bracket_bar_h,
			refine_max_shift, refine_pad_v, refine_pad_h,
			row_pitch, col_pitch
		"""
		rp = self._row_pitch
		cp = self._col_pitch
		cfg = {
			"center_exclusion": _DZ_CENTER_EXCLUSION * cp,
			"bracket_inner_half": _DZ_BRACKET_INNER_HALF * cp,
			"fill_inset_v": _DZ_FILL_INSET_V * rp,
			"bracket_bar_v": _DZ_BRACKET_BAR_V * rp,
			"bracket_bar_h": _DZ_BRACKET_BAR_H * rp,
			"refine_max_shift": _DZ_REFINE_MAX_SHIFT * rp,
			"refine_pad_v": _DZ_REFINE_PAD_V * rp,
			"refine_pad_h": _DZ_REFINE_PAD_H * cp,
			"row_pitch": rp,
			"col_pitch": cp,
		}
		return cfg

	#============================================
	def sid_roi_bounds(self, digit_idx: int, value: int) -> tuple:
		"""Return (top_y, bot_y, left_x, right_x) for a student ID bubble.

		Uses lattice midpoints between neighboring centers, same pattern
		as roi_bounds(). Primary geometry method for student ID scoring.

		Args:
			digit_idx: digit position (0 to num_digits-1)
			value: digit value (0-9)

		Returns:
			tuple of (top_y, bot_y, left_x, right_x) as integers
		"""
		num_rows = len(self._sid_row_y)
		num_cols = len(self._sid_col_x)
		# --- vertical bounds from neighboring row centers ---
		if 0 < value < num_rows - 1:
			# interior row: midpoint to neighbors
			top_y = int(round(
				(self._sid_row_y[value - 1] + self._sid_row_y[value]) / 2.0))
			bot_y = int(round(
				(self._sid_row_y[value] + self._sid_row_y[value + 1]) / 2.0))
		elif value == 0:
			# first row: extrapolate top by half pitch
			top_y = int(round(
				self._sid_row_y[0] - self._sid_row_pitch / 2.0))
			bot_y = int(round(
				(self._sid_row_y[0] + self._sid_row_y[1]) / 2.0))
		else:
			# last row: extrapolate bottom by half pitch
			top_y = int(round(
				(self._sid_row_y[-2] + self._sid_row_y[-1]) / 2.0))
			bot_y = int(round(
				self._sid_row_y[-1] + self._sid_row_pitch / 2.0))
		# --- horizontal bounds from neighboring column centers ---
		d = digit_idx
		if 0 < d < num_cols - 1:
			# interior column: midpoint to neighbors
			left_x = int(round(
				(self._sid_col_x[d - 1] + self._sid_col_x[d]) / 2.0))
			right_x = int(round(
				(self._sid_col_x[d] + self._sid_col_x[d + 1]) / 2.0))
		elif d == 0:
			# first column: extrapolate left by half col_pitch
			left_x = int(round(
				self._sid_col_x[0] - self._col_pitch / 2.0))
			right_x = int(round(
				(self._sid_col_x[0] + self._sid_col_x[1]) / 2.0))
		else:
			# last column: extrapolate right by half col_pitch
			left_x = int(round(
				(self._sid_col_x[-2] + self._sid_col_x[-1]) / 2.0))
			right_x = int(round(
				self._sid_col_x[-1] + self._col_pitch / 2.0))
		return (top_y, bot_y, left_x, right_x)

	#============================================
	def sid_center(self, digit_idx: int, value: int) -> tuple:
		"""Return (cx, cy) pixel coords for a student ID bubble.

		For debug overlay and labeling only. Not used in scoring.

		Args:
			digit_idx: digit position (0 to num_digits-1)
			value: digit value (0-9)

		Returns:
			tuple of (cx, cy) integer pixel coords
		"""
		cx = int(round(self._sid_col_x[digit_idx]))
		cy = int(round(self._sid_row_y[value]))
		return (cx, cy)
