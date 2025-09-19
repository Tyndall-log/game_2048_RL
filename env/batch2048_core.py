# batch2048_core.py
from __future__ import annotations
import enum

import numpy as np


class Batch2048Core:
	"""
	벡터화 2048 순수 코어

	- 내부 상태: boards (N, 4) uint16, 각 원소는 4칸(=4니블, 상위→하위 칸)
		- 한 행(16비트)에 4칸의 지수 e(0..15)를 담는다. 실제 타일 값은 2^e, e=0은 빈칸.
	- 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
	- 보상: 2048 룰(두 타일 e,e가 병합되어 e+1 생성 시 2^(e+1) 점수) 합
	- 스폰: 이동이 발생한 보드에만 1개 타일 스폰 (2: 확률 1-p4, 4: 확률 p4)
	- obs_mode:
		- "uint16x4" : (N, 4) uint16, 내부 보드 그대로
		- "uint8x16" : (N, 16) uint8, 16칸의 지수값 벡터
		- "onehot256": (N, 256) uint8/float32, 16칸×16클래스 원-핫 (flatten)
	"""

	# -------- 클래스 정적 LUT들 (최초 1회 생성) --------
	_LUT_LEFT_NEW: np.ndarray | None = None        # uint16[65536], LEFT 적용 결과 행
	_LUT_RIGHT_NEW: np.ndarray | None = None       # uint16[65536], RIGHT 적용 결과 행
	_LUT_LEFT_MOV: np.ndarray | None = None        # bool[65536],  LEFT 적용 시 변화 여부
	_LUT_RIGHT_MOV: np.ndarray | None = None       # bool[65536],  RIGHT 적용 시 변화 여부
	_LUT_LR_NEW: np.ndarray | None = None          # uint16[65536, 2] (0=left,1=right)
	_LUT_LEFT_REW: np.ndarray | None = None        # uint32[65536], LEFT 보상 합
	_LUT_RIGHT_REW: np.ndarray | None = None       # uint32[65536], RIGHT 보상 합

	# 스폰 최적화용 LUT들
	_PC4: np.ndarray | None = None                 # uint8[16],     4비트 popcount
	_PC16: np.ndarray | None = None                # uint16[65536], 16비트 popcount
	_LUT_EMPTY4_ROW: np.ndarray | None = None      # uint16[65536], row16 -> empty 4bit mask
	_LUT_MASK_SELECT: np.ndarray | None = None     # uint8[16,4],   (mask4, nth) -> col or 255
	_LUT_SELECT16_ROWS: np.ndarray | None = None   # uint16[65536,16], (mask16,nth)->row or 255
	_LUT_SELECT16_COLS_REVERSE: np.ndarray | None = None  # uint16[65536,16], (mask16,nth)->col or 255

	class ObsMode(enum.IntEnum):
		UINT16x4 = enum.auto()
		UINT8x16 = enum.auto()
		ONEHOT256 = enum.auto()

	def __init__(
		self,
		obs_mode: ObsMode = ObsMode.UINT8x16,
		num_envs: int = 1024,
		seed: int | None = None,
		p4: float = 0.1,
	):
		self.num_envs = int(num_envs)
		self.obs_mode = obs_mode
		self._obs_func = self._select_obs_fn(obs_mode)
		self._rng = np.random.default_rng(seed)
		self._boards = np.zeros((self.num_envs, 4), dtype=np.uint16)
		self._boards_T = np.zeros((self.num_envs, 4), dtype=np.uint16)
		self._last_legal_flags = np.zeros((self.num_envs, 4), dtype=bool)
		self._p4 = float(p4)

		# 좌/우 LUT 준비 (최초 1회)
		if Batch2048Core._LUT_LEFT_NEW is None:
			lut_left_new, lut_right_new, lut_left_rew, lut_right_rew = self._build_row_luts()
			Batch2048Core._LUT_LEFT_NEW = lut_left_new
			Batch2048Core._LUT_RIGHT_NEW = lut_right_new
			Batch2048Core._LUT_LEFT_REW = lut_left_rew
			Batch2048Core._LUT_RIGHT_REW = lut_right_rew
			Batch2048Core._LUT_LR_NEW = np.stack(
				[
					Batch2048Core._LUT_LEFT_NEW,
					Batch2048Core._LUT_RIGHT_NEW,
				],
				axis=1,
			)
			base = np.arange(65536, dtype=np.uint16)
			Batch2048Core._LUT_LEFT_MOV = (Batch2048Core._LUT_LEFT_NEW != base)
			Batch2048Core._LUT_RIGHT_MOV = (Batch2048Core._LUT_RIGHT_NEW != base)

		# 스폰 LUT 준비 (최초 1회)
		if Batch2048Core._LUT_EMPTY4_ROW is None:
			self._init_spawn_luts()

	# ---------------- 공개 API ----------------

	def reset(
		self, *, seed: int | None = None,
		mask: np.ndarray | None = None,
		indices: np.ndarray | list | tuple | None = None,
	):
		"""
		Arg:
			seed: RNG 재시드용 시드값 (선택적)
			mask: (N,) bool 배열, True인 env만 리셋 (권장)
			indices: 리셋할 인덱스들 (레거시, mask 사용 권장)

		Returns:
			obs: obs_mode에 따른 관측 (N, ...)
			info: {"reset_mask": (N,) bool, "legal_actions": (N,4) bool}
		"""
		if seed is not None:
			self._rng = np.random.default_rng(seed)

		if (mask is not None) and (indices is not None):
			raise ValueError("Provide either mask or indices, not both.")

		# 1) mask 정규화
		if mask is None:
			if indices is None:
				# 전체 리셋
				reset_mask = np.ones((self.num_envs,), dtype=bool)
			else:
				idx = np.asarray(indices)
				if idx.dtype == np.bool_:
					if idx.shape != (self.num_envs,):
						raise ValueError(f"indices(bool) shape must be ({self.num_envs},)")
					reset_mask = idx.astype(bool, copy=False)
				else:
					reset_mask = np.zeros((self.num_envs,), dtype=bool)
					reset_mask[idx.astype(np.int64, copy=False)] = True
		else:
			if mask.dtype != np.bool_ or mask.shape != (self.num_envs,):
				raise ValueError(f"mask must be bool array of shape ({self.num_envs},)")
			reset_mask = mask

		# 2) 보드 초기화 + 스폰 (reset_mask=True인 곳만)
		if reset_mask.any():
			self._boards[reset_mask] = 0
			self._spawn_random_tile_batch_bitwise(self._boards, moved_mask=reset_mask, p4=self._p4)

		# 3) 합법 액션 플래그/정보
		canL, canR = self._compute_action_flags(self._boards)
		self._transpose_all(self._boards, out=self._boards_T)
		canU, canD = self._compute_action_flags(self._boards_T)
		legal_flags = np.stack([canL, canR, canU, canD], axis=1)

		info = {
			"reset_mask": reset_mask,  # (N,)
			"legal_actions": legal_flags,  # (N,4)
		}
		return self._obs_func(), info

	def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
		"""
		Arg:
			actions: (N,) int64 배열, 값은 {0,1,2,3} (0=LEFT,1=RIGHT,2=UP,3=DOWN)

		Returns:
			obs: (N, ...) obs_mode에 따른 관측
			reward: (N,) float32 보상
			terminated: (N,) bool 합법 액션이 없을 때 True
			truncated: (N,) bool 항상 False (상위에서 time-limit 처리)
			info: dict{"invalid_move": (N,) bool, "legal_actions": (N,4) bool}
		"""
		actions = np.asarray(actions)
		if actions.shape != (self.num_envs,):
			raise ValueError(f"actions must have shape ({self.num_envs},), got {actions.shape}")

		moved_mask = np.zeros((self.num_envs,), dtype=bool)
		reward_sum = np.zeros((self.num_envs,), dtype=np.int64)

		# 수평 액션 (LEFT/RIGHT)
		idx_h = np.nonzero((actions == 0) | (actions == 1))[0]
		if idx_h.size:
			idx_left = idx_h[actions[idx_h] == 0]
			if idx_left.size:
				moved, rew = self._apply_lut_inplace(
					self._boards, idx_left,
					Batch2048Core._LUT_LEFT_NEW,
					Batch2048Core._LUT_LEFT_MOV,
					Batch2048Core._LUT_LEFT_REW,
				)
				moved_mask[idx_left] = moved
				reward_sum[idx_left] += rew

			idx_right = idx_h[actions[idx_h] == 1]
			if idx_right.size:
				moved, rew = self._apply_lut_inplace(
					self._boards, idx_right,
					Batch2048Core._LUT_RIGHT_NEW,
					Batch2048Core._LUT_RIGHT_MOV,
					Batch2048Core._LUT_RIGHT_REW,
				)
				moved_mask[idx_right] = moved
				reward_sum[idx_right] += rew

		# 수직 액션 (UP/DOWN): 전치 보드에서 좌/우 LUT 적용
		self._transpose_all(self._boards, out=self._boards_T)
		idx_v = np.nonzero((actions == 2) | (actions == 3))[0]
		if idx_v.size:
			idx_up = idx_v[actions[idx_v] == 2]
			if idx_up.size:
				moved, rew = self._apply_lut_inplace(
					self._boards_T, idx_up,
					Batch2048Core._LUT_LEFT_NEW,
					Batch2048Core._LUT_LEFT_MOV,
					Batch2048Core._LUT_LEFT_REW,
				)
				moved_mask[idx_up] = moved
				reward_sum[idx_up] += rew

			idx_down = idx_v[actions[idx_v] == 3]
			if idx_down.size:
				moved, rew = self._apply_lut_inplace(
					self._boards_T, idx_down,
					Batch2048Core._LUT_RIGHT_NEW,
					Batch2048Core._LUT_RIGHT_MOV,
					Batch2048Core._LUT_RIGHT_REW,
				)
				moved_mask[idx_down] = moved
				reward_sum[idx_down] += rew

		# 이동된 보드에만 타일 스폰 (전치 상태에서 보드가 최신)
		self._spawn_random_tile_batch_bitwise(self._boards_T, moved_mask, p4=self._p4)

		# 다음 상태의 합법 액션 플래그 & 종료 판정
		canU, canD = self._compute_action_flags(self._boards_T)
		self._transpose_all(self._boards_T, out=self._boards)
		canL, canR = self._compute_action_flags(self._boards)
		legal_flags = np.stack([canL, canR, canU, canD], axis=1)
		terminated = ~legal_flags.any(axis=1)
		self._last_legal_flags = legal_flags

		obs = self._obs_func()
		reward = reward_sum.astype(np.float32, copy=False)
		truncated = np.zeros((self.num_envs,), dtype=np.bool_)
		info = {
			"invalid_move": ~moved_mask,   # 이번 액션에서 실제로 아무 변화도 없었는가
			"legal_actions": legal_flags,  # 다음 스텝에서 가능한 [L, R, U, D]
		}
		return obs, reward, terminated, truncated, info

	# ---------------- 관측 변환 ----------------
	@staticmethod
	def uint16x4_to_uint8x16(boards: np.ndarray) -> np.ndarray:
		"""
		Arg:
			boards: (N,4) uint16 배열, 각 원소는 4칸의 지수값을 담은 16비트

		Returns:
			(N,16) uint8 배열, 각 칸의 지수 e 값
		"""
		b = boards
		c0 = ((b >> 12) & 0xF)
		c1 = ((b >> 8) & 0xF)
		c2 = ((b >> 4) & 0xF)
		c3 = (b & 0xF)
		return np.stack([c0, c1, c2, c3], axis=2).astype(np.uint8).reshape(b.shape[0], 16)

	def get_original_boards(self) -> np.ndarray:
		"""
		Returns:
			내부 보드 상태 복사본 (N,4) uint16
		"""
		return self._boards.copy()

	def boards_as_uint8x16(self) -> np.ndarray:
		"""
		Returns:
			(N,16) uint8 배열, 각 칸의 지수 e 값
		"""
		b = self._boards
		c0 = ((b >> 12) & 0xF)
		c1 = ((b >> 8) & 0xF)
		c2 = ((b >> 4) & 0xF)
		c3 = (b & 0xF)
		return np.stack([c0, c1, c2, c3], axis=2).astype(np.uint8).reshape(b.shape[0], 16)

	def boards_onehot256(self, *, dtype=np.uint8, flatten: bool = True) -> np.ndarray:
		"""
		Arg:
			dtype: 원-핫 인코딩에 사용할 데이터 타입 (기본값: np.uint8)
			flatten: True면 (N,256), False면 (N,16,16) 형태로 반환

		Returns:
			원-핫 인코딩된 배열 (N,256) 또는 (N,16,16)
		"""
		b = self._boards
		c0 = ((b >> 12) & 0xF)
		c1 = ((b >> 8) & 0xF)
		c2 = ((b >> 4) & 0xF)
		c3 = (b & 0xF)
		vals = np.stack([c0, c1, c2, c3], axis=2).reshape(b.shape[0], 16)  # (N,16)
		eye16 = np.eye(16, dtype=dtype)
		onehot = eye16[vals]  # (N,16,16)
		return onehot.reshape(b.shape[0], 16 * 16) if flatten else onehot

	# ---------------- 보조 메트릭 ----------------

	def best_tile(self) -> np.ndarray:
		"""
		Returns:
			각 보드에서 가장 큰 타일의 지수값 (N,) uint8
		"""
		b = self._boards
		c0 = ((b >> 12) & 0xF)
		c1 = ((b >> 8) & 0xF)
		c2 = ((b >> 4) & 0xF)
		c3 = (b & 0xF)
		row_max = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3)).astype(np.uint8)
		return row_max.max(axis=1)

	def tile_score_sum(self) -> np.ndarray:
		"""현재 보드의 타일 값 합 Σ 2^e (빈칸=0) → (N,)"""
		b = self._boards
		c0 = ((b >> 12) & 0xF)
		c1 = ((b >> 8) & 0xF)
		c2 = ((b >> 4) & 0xF)
		c3 = (b & 0xF)
		v0 = np.where(c0 > 0, 1 << c0, 0)
		v1 = np.where(c1 > 0, 1 << c1, 0)
		v2 = np.where(c2 > 0, 1 << c2, 0)
		v3 = np.where(c3 > 0, 1 << c3, 0)
		return (v0 + v1 + v2 + v3).sum(axis=1)

	def estimated_cumulative_score(self, *, out_dtype=np.int64) -> np.ndarray:
		"""
		Arg:
			out_dtype: 출력 데이터 타입 (기본값: np.int64)

		Returns:
			누적 점수 근사값 Σ_{e>0} 2^e * (e-1) (N,) 배열
			스폰이 항상 2였다고 가정 시 정확, 4가 섞이면 약간 과대추정
		"""
		b = self._boards
		e0 = ((b >> 12) & 0xF).astype(np.int64)
		e1 = ((b >> 8) & 0xF).astype(np.int64)
		e2 = ((b >> 4) & 0xF).astype(np.int64)
		e3 = (b & 0xF).astype(np.int64)

		def contrib(e: np.ndarray) -> np.ndarray:
			return np.where(e > 0, (np.int64(1) << e) * (e - 1), 0)

		total = (contrib(e0) + contrib(e1) + contrib(e2) + contrib(e3)).sum(axis=1)
		return total.astype(out_dtype, copy=False)

	# ---------------- 내부 유틸/루틴 ----------------

	@staticmethod
	def _pack_row(vals: np.ndarray) -> int:
		# vals: (4,) uint8  [a b c d] (a가 상위니블)
		return (int(vals[0]) << 12) | (int(vals[1]) << 8) | (int(vals[2]) << 4) | int(vals[3])

	@staticmethod
	def _unpack_row(r: int) -> np.ndarray:
		return np.array([(r >> 12) & 0xF, (r >> 8) & 0xF, (r >> 4) & 0xF, r & 0xF], dtype=np.uint8)

	@classmethod
	def _slide_merge_left_row(cls, vals: np.ndarray) -> np.uint16:
		# 보상 계산 없이 LEFT 결과 행만
		comp = [int(v) for v in vals if v != 0]
		out = []
		i = 0
		while i < len(comp):
			if i + 1 < len(comp) and comp[i] == comp[i + 1]:
				out.append(comp[i] + 1); i += 2
			else:
				out.append(comp[i]); i += 1
		while len(out) < 4:
			out.append(0)
		return np.uint16(cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15)))

	@classmethod
	def _slide_merge_left_row_with_reward(cls, vals: np.ndarray) -> tuple[np.uint16, int]:
		# LEFT로 슬라이드+머지: (새 행, 이번 이동 보상)
		comp = [int(v) for v in vals if v != 0]
		out = []
		i = 0
		reward = 0
		while i < len(comp):
			if i + 1 < len(comp) and comp[i] == comp[i + 1]:
				new_e = comp[i] + 1
				out.append(new_e)
				reward += (1 << new_e)  # 2048 점수 룰
				i += 2
			else:
				out.append(comp[i]); i += 1
		while len(out) < 4:
			out.append(0)
		row16 = np.uint16(cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15)))
		return row16, reward

	@classmethod
	def _build_row_luts(cls):
		"""좌/우 결과행 및 보상 LUT 생성"""
		lut_left = np.zeros(65536, dtype=np.uint16)
		lut_right = np.zeros(65536, dtype=np.uint16)
		lut_left_rew = np.zeros(65536, dtype=np.uint32)
		lut_right_rew = np.zeros(65536, dtype=np.uint32)

		def reverse_row16(r: int) -> int:
			# abcd -> dcba
			return ((r & 0x000F) << 12) | ((r & 0x00F0) << 4) | ((r & 0x0F00) >> 4) | ((r & 0xF000) >> 12)

		for r in range(65536):
			orig = cls._unpack_row(r)
			left_r, left_rew = cls._slide_merge_left_row_with_reward(orig)
			lut_left[r] = left_r
			lut_left_rew[r] = left_rew

			rev = reverse_row16(r)
			rev_orig = cls._unpack_row(rev)
			rev_left_r, rev_left_rew = cls._slide_merge_left_row_with_reward(rev_orig)
			right_r = reverse_row16(int(rev_left_r))
			lut_right[r] = right_r
			lut_right_rew[r] = rev_left_rew

		return lut_left, lut_right, lut_left_rew, lut_right_rew

	@staticmethod
	def obs_shape(mode: ObsMode) -> tuple[int, ...]:
		if mode is Batch2048Core.ObsMode.UINT16x4:   return (4,)
		if mode is Batch2048Core.ObsMode.UINT8x16:   return (16,)
		if mode is Batch2048Core.ObsMode.ONEHOT256:  return (256,)
		raise ValueError(mode)

	@staticmethod
	def obs_dtype(mode: ObsMode):
		if mode is Batch2048Core.ObsMode.UINT16x4:   return np.uint16
		if mode is Batch2048Core.ObsMode.UINT8x16:   return np.uint8
		if mode is Batch2048Core.ObsMode.ONEHOT256:  return np.uint8
		raise ValueError(mode)

	def _select_obs_fn(self, mode: ObsMode):
		if mode is self.ObsMode.UINT16x4:
			return lambda: self._boards.copy()
		if mode is self.ObsMode.UINT8x16:
			return self.boards_as_uint8x16
		if mode is self.ObsMode.ONEHOT256:
			return self.boards_onehot256
		raise ValueError(f"Unsupported obs_mode: {mode}")

	def _apply_lut_inplace(
		self,
		target_board: np.ndarray,
		idx: np.ndarray,
		lut_rows: np.ndarray,
		lut_moved: np.ndarray,
		lut_rewards: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""선택된 보드에 좌/우 LUT 적용 및 보상 합산"""
		boards = target_board
		b0 = boards[idx, 0]; b1 = boards[idx, 1]; b2 = boards[idx, 2]; b3 = boards[idx, 3]

		moved_any = (lut_moved[b0] | lut_moved[b1] | lut_moved[b2] | lut_moved[b3])
		rewards = (
			lut_rewards[b0].astype(np.int64) +
			lut_rewards[b1].astype(np.int64) +
			lut_rewards[b2].astype(np.int64) +
			lut_rewards[b3].astype(np.int64)
		)

		boards[idx, 0] = lut_rows[b0]
		boards[idx, 1] = lut_rows[b1]
		boards[idx, 2] = lut_rows[b2]
		boards[idx, 3] = lut_rows[b3]

		return moved_any, rewards

	def _transpose_all(self, x: np.ndarray, out: np.ndarray):
		"""(N,4) 행-니블 보드를 니블 전치하여 out에 저장"""
		a = x[:, 0]; b = x[:, 1]; c = x[:, 2]; d = x[:, 3]
		t0 = (a & 0xF000) | ((b & 0xF000) >> 4) | ((c & 0xF000) >> 8) | ((d & 0xF000) >> 12)
		t1 = ((a & 0x0F00) << 4) | (b & 0x0F00) | ((c & 0x0F00) >> 4) | ((d & 0x0F00) >> 8)
		t2 = ((a & 0x00F0) << 8) | ((b & 0x00F0) << 4) | (c & 0x00F0) | ((d & 0x00F0) >> 4)
		t3 = ((a & 0x000F) << 12) | ((b & 0x000F) << 8) | ((c & 0x000F) << 4) | (d & 0x000F)
		out[:, 0] = t0; out[:, 1] = t1; out[:, 2] = t2; out[:, 3] = t3

	def _compute_action_flags(self, target_board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""현재 보드에서 (LEFT 가능?, RIGHT 가능?) 플래그 튜플 반환"""
		lut_L = Batch2048Core._LUT_LEFT_MOV
		lut_R = Batch2048Core._LUT_RIGHT_MOV
		b0 = target_board[:, 0]; b1 = target_board[:, 1]; b2 = target_board[:, 2]; b3 = target_board[:, 3]
		canL = (lut_L[b0] | lut_L[b1] | lut_L[b2] | lut_L[b3])
		canR = (lut_R[b0] | lut_R[b1] | lut_R[b2] | lut_R[b3])
		return canL, canR

	def _init_spawn_luts(self):
		"""스폰 최적화용 LUT 일괄 생성(최초 1회)"""
		pc4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)

		lut_sel4 = np.full((16, 4), 255, dtype=np.uint8)
		for mask in range(16):
			cols = []
			for col in range(4):  # col=0..3 (왼→오)
				bit = 3 - col  # bit3↔col0, bit0↔col3
				if (mask >> bit) & 1:
					cols.append(col)
			for n, col in enumerate(cols):
				lut_sel4[mask, n] = col

		empty4 = np.zeros(65536, dtype=np.uint16)
		for r in range(65536):
			m3 = 1 if ((r & 0xF000) == 0) else 0
			m2 = 1 if ((r & 0x0F00) == 0) else 0
			m1 = 1 if ((r & 0x00F0) == 0) else 0
			m0 = 1 if ((r & 0x000F) == 0) else 0
			empty4[r] = (m3 << 3) | (m2 << 2) | (m1 << 1) | m0

		pc16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint16)

		lut_sel16_row = np.full((1 << 16, 16), 255, dtype=np.uint16)
		lut_sel16_col = np.full((1 << 16, 16), 255, dtype=np.uint16)
		for m in range(1 << 16):
			m0 = (m >> 12) & 0xF
			m1 = (m >> 8) & 0xF
			m2 = (m >> 4) & 0xF
			m3 = (m >> 0) & 0xF
			c0 = int(pc4[m0]); c1 = int(pc4[m1]); c2 = int(pc4[m2]); c3 = int(pc4[m3])

			for n in range(c0):
				col = lut_sel4[m0, n]
				lut_sel16_row[m, n] = 0
				lut_sel16_col[m, n] = 3 - col
			base = c0
			for n in range(c1):
				col = lut_sel4[m1, n]
				lut_sel16_row[m, base + n] = 1
				lut_sel16_col[m, base + n] = 3 - col
			base += c1
			for n in range(c2):
				col = lut_sel4[m2, n]
				lut_sel16_row[m, base + n] = 2
				lut_sel16_col[m, base + n] = 3 - col
			base += c2
			for n in range(c3):
				col = lut_sel4[m3, n]
				lut_sel16_row[m, base + n] = 3
				lut_sel16_col[m, base + n] = 3 - col

		Batch2048Core._PC4 = pc4
		Batch2048Core._PC16 = pc16
		Batch2048Core._LUT_EMPTY4_ROW = empty4
		Batch2048Core._LUT_MASK_SELECT = lut_sel4
		Batch2048Core._LUT_SELECT16_ROWS = lut_sel16_row
		Batch2048Core._LUT_SELECT16_COLS_REVERSE = lut_sel16_col

	def _spawn_random_tile_batch_bitwise(self, target_board: np.ndarray, moved_mask: np.ndarray, p4: float = 0.1):
		"""
		이동이 있었던 보드들에만 타일 1개 스폰.
		- 빈칸 보드 마스크(16비트)를 LUT로 계산, 무작위 nth 빈칸을 골라 2/4 배치.
		"""
		idx_env = np.nonzero(moved_mask)[0]
		if idx_env.size == 0:
			return

		empty4 = Batch2048Core._LUT_EMPTY4_ROW
		pc16 = Batch2048Core._PC16

		row_masks = empty4[target_board[idx_env]]  # (M,4)
		board_mask16 = ((row_masks[:, 0] << 12) | (row_masks[:, 1] << 8) | (row_masks[:, 2] << 4) | (row_masks[:, 3] << 0))

		total_empty = pc16[board_mask16]  # (M,)
		valid = total_empty > 0
		if not np.any(valid):
			return

		env_ids = idx_env[valid]
		v_mask16 = board_mask16[valid]
		v_tot = total_empty[valid]

		rng = self._rng
		v_nth = rng.integers(0, v_tot, dtype=np.uint16)  # (Mv,)
		v_k = np.where(rng.random(size=v_tot.shape) < p4, 2, 1).astype(np.uint16)

		rows = Batch2048Core._LUT_SELECT16_ROWS[v_mask16, v_nth]
		cols = Batch2048Core._LUT_SELECT16_COLS_REVERSE[v_mask16, v_nth]
		shift = (cols << 2)  # 0,4,8,12

		target_board[env_ids, rows] |= (v_k << shift)

	# ---------------- 디버그용 ----------------

	# @staticmethod
	# def render_board_text(board_row16: np.ndarray) -> str:
	# 	"""
	# 	(4,) uint16 보드를 텍스트 그리드로 변환 (디버그 출력용)
	# 	각 칸은 실제 값(2^e)로 표시, 빈칸은 0.
	# 	"""
	# 	result = ""
	# 	for r in board_row16:
	# 		cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
	# 		result += " ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells) + "\n"
	# 	return result

	@staticmethod
	def render_obs(obs: np.ndarray) -> str:
		"""
		obs_mode에 따른 관측을 텍스트 그리드로 변환 (디버그 출력용)
		각 칸은 실제 값(2^e)로 표시, 빈칸은 0.
		지원: (N,4)[uint16], (N,16)[uint8], (N,256)[uint8|float32]
		"""
		obs_channel = obs.shape[-1]
		if obs.ndim != 2 or obs_channel not in (4, 16, 256):
			raise ValueError("render_obs only supports obs with shape (N,4), (N,16), or (N,256)")

		fmt = lambda e: f"{((1 << int(e)) if int(e) > 0 else 0):^5d}"
		lines = []

		if obs_channel == 4:
			board_Nx4x4 = Batch2048Core.uint16x4_to_uint8x16(obs).reshape(-1, 4, 4)
		elif obs_channel == 16:
			board_Nx4x4 = obs.reshape(-1, 4, 4)
		elif obs_channel == 256:
			grid = obs.reshape(-1, 4, 4, 16)
			exps = np.argmax(grid, axis=-1)  # (N,4,4)
			board_Nx4x4 = exps
		else:
			raise ValueError("Unsupported obs format in render_obs")

		for row in board_Nx4x4.swapaxes(0, 1):
			parts = [" ".join(fmt(e) for e in r) for r in row]
			lines.append(" | ".join(parts))
		return "\n".join(lines)


# if __name__ == "__main__":
# 	import tqdm  # 속도 체크용
# 	# 간단한 스모크 테스트 (SB3 없이 코어 단독 구동)
# 	core = Batch2048Core(obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=2**15, seed=42)
# 	obs, info = core.reset()
# 	print("Initial legal ratio:", info["legal_actions"].mean())
#
# 	total_steps = 1000 * (2**18 // core.num_envs)
# 	for _ in tqdm.tqdm(range(total_steps)):
# 		# 무작위 액션 (N,)
# 		actions = core._rng.integers(0, 4, size=core.num_envs, dtype=np.int64)
# 		obs, reward, terminated, truncated, info = core.step(actions)
#
# 		# 종료된 env만 부분 리셋
# 		if np.any(terminated):
# 			core.reset(mask=terminated)
#
# 	print("Mean best tile:", core.best_tile().mean())
# 	print("Mean tile sum:", core.tile_score_sum().mean())
# 	print("Mean score est.:", core.estimated_cumulative_score().mean())


if __name__ == "__main__":
	env = Batch2048Core(obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=2, seed=42)
	obs, info = env.reset()
	print("Initial boards:")
	print(env.render_obs(obs))
	print("Info:", info)

	for step in range(5000):
		input_str = input("Enter action (a=LEFT, d=RIGHT, w=UP, s=DOWN, q=quit): ").strip().lower()
		if input_str == 'q':
			break
		action_map = {'a': 0, 'd': 1, 'w': 2, 's': 3}
		if input_str not in action_map:
			print("Invalid input. Please enter a, d, w, s, or q.")
			continue
		actions = np.array([action_map[input_str]] * env.num_envs, dtype=np.int64)
		# actions = env._rng.integers(0, 4, size=env.num_envs, dtype=np.int64)  # 무작위 액션
		obs, reward, terminated, truncated, info = env.step(actions)
		obs, info2 = env.reset(mask=terminated)
		print(f"\nStep {step+1}, Actions: {actions}")
		print(f"Info_legal: {info['legal_actions']}, Info_invalid: {info['invalid_move']}")
		print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
		print("Boards:")
		print(env.render_obs(obs))
		print("Best tiles:", env.best_tile())
		print("Sum tiles:", env.tile_score_sum())

