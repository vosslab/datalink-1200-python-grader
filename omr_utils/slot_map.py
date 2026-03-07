"""SlotMap: single source of truth for bubble slot geometry.

Builds pixel coordinates directly from timing mark anchors.
Row centers come from left question marks, column centers from
top footprint origin + choice_columns * local lattice column pitch.
No YAML coordinates, no affine correction, no Sobel refinement.
"""

# Standard Library
import math


# Dimensionless fractions relating bubble geometry to timing-mark spacing.
# Each K constant is the ratio of a bubble measurement to the
# corresponding pitch (row_pitch for vertical, col_pitch for horizontal).
# Estimated from curated scans with correct bracket overlay alignment.
_K_CENTER_EXCLUSION = 0.244    # center letter zone / col_pitch
_K_BRACKET_EDGE_H = 0.043     # bracket edge height / row_pitch
_K_MEAS_INSET_V = 0.043       # vertical measurement inset / row_pitch
_K_MEAS_INSET_H = 0.067       # horizontal measurement inset / col_pitch
_K_REFINE_MAX_SHIFT = 0.321   # max template shift / row_pitch
_K_REFINE_PAD_V = 0.171       # vertical refine padding / row_pitch
_K_REFINE_PAD_H = 0.177       # horizontal refine padding / col_pitch


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
		fine_step = float(transform.get("top_col_spacing", 0.0))
		if fp_x0 <= 0 or fine_step <= 0:
			raise ValueError(
				f"SlotMap: top footprint unavailable "
				f"(fp_x0={fp_x0}, fine_step={fine_step})")
		question_marks = transform.get("left_question_marks", [])
		if len(question_marks) < 50:
			raise ValueError(
				f"SlotMap: need 50 question marks, "
				f"got {len(question_marks)}")
		# store pitches
		self._row_pitch = float(transform.get("left_s_q", 0.0))
		self._col_pitch = float(fine_step)
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
			self._col_x[("left", choice)] = fp_x0 + left_cols[choice] * fine_step
			self._col_x[("right", choice)] = fp_x0 + right_cols[choice] * fine_step

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
	def measure_cfg(self) -> dict:
		"""Return measurement constants derived from timing-mark spacing.

		Provides dimensions for measuring within a bubble (Sobel search
		extents, center exclusion, insets, validation thresholds).
		Use SlotMap.roi_bounds() and SlotMap.center() for slot placement.

		Returns:
			dict with keys: center_exclusion, bracket_edge_height,
			measurement_inset_v, measurement_inset_h, refine_max_shift,
			refine_pad_v, refine_pad_h, row_pitch, col_pitch
		"""
		rp = self._row_pitch
		cp = self._col_pitch
		cfg = {
			"center_exclusion": _K_CENTER_EXCLUSION * cp,
			"bracket_edge_height": _K_BRACKET_EDGE_H * rp,
			"measurement_inset_v": _K_MEAS_INSET_V * rp,
			"measurement_inset_h": _K_MEAS_INSET_H * cp,
			"refine_max_shift": _K_REFINE_MAX_SHIFT * rp,
			"refine_pad_v": _K_REFINE_PAD_V * rp,
			"refine_pad_h": _K_REFINE_PAD_H * cp,
			"row_pitch": rp,
			"col_pitch": cp,
		}
		return cfg

	#============================================
	def student_id_geom(self) -> dict:
		"""Return pixel geometry dict for student ID bubble scoring.

		Student-ID subsystem only. Extends measure_cfg() with half_width
		and half_height needed by score_bubble_fast(). Answer-bubble code
		should use measure_cfg() instead.

		Returns:
			dict with all measure_cfg keys plus half_width, half_height
		"""
		rp = self._row_pitch
		cp = self._col_pitch
		result = self.measure_cfg()
		# dimensionless fractions for student-ID center-plus-box model
		result["half_width"] = 0.665 * cp
		result["half_height"] = 0.1175 * rp
		return result
