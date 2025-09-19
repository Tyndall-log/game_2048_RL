# adapters_2048.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
	from stable_baselines3.common.vec_env import VecEnv
except Exception as e:
	VecEnv = object  # SB3 미사용 환경에서도 import 에러 방지용 더미

from env.batch2048_core import Batch2048Core


def _single_obs_space_for(mode: Batch2048Core.ObsMode):
	shape = Batch2048Core.obs_shape(mode)
	dtype = Batch2048Core.obs_dtype(mode)
	# SB3는 단일 env 기준 space 필요
	if mode is Batch2048Core.ObsMode.UINT16x4:
		return spaces.Box(low=0, high=np.uint16(0xFFFF), shape=shape, dtype=dtype)
	if mode is Batch2048Core.ObsMode.UINT8x16:
		return spaces.Box(low=0, high=15, shape=shape, dtype=dtype)  # 지수 e 범위
	if mode is Batch2048Core.ObsMode.ONEHOT256:
		return spaces.Box(low=0, high=1, shape=shape, dtype=dtype)
	raise ValueError(mode)


# =========================
# 1) Gym 단일 환경 어댑터
# =========================

class Gym2048SingleEnv(gym.Env):
	"""
	Gym 단일 환경 어댑터.
	- 내부적으로 Batch2048Core(num_envs=1)를 사용해 표준 Gym API를 제공.
	- 관측/행동 space는 단일 환경 기준으로 노출.
	- 시간 제한 같은 트렁케이션은 상위에서 래핑해 사용.
	"""
	metadata = {"render_modes": ["ansi"]}

	def __init__(
		self,
		obs_mode: Batch2048Core.ObsMode = Batch2048Core.ObsMode.UINT8x16,
		seed: int | None = None,
		p4: float = 0.1,
	):
		super().__init__()
		self.obs_mode = obs_mode
		self.core = Batch2048Core(obs_mode=obs_mode, num_envs=1, seed=seed, p4=p4)

		# 단일 환경 기준 space
		self.observation_space = _single_obs_space_for(obs_mode)
		self.action_space = spaces.Discrete(4)  # 0=LEFT,1=RIGHT,2=UP,3=DOWN

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		if seed is not None:
			obs, info = self.core.reset(seed=seed)
		else:
			obs, info = self.core.reset()
		# core는 (N, …) 반환 → 단일 env이므로 [0]
		return obs[0], self._slice_info(info, 0)

	def step(self, action):
		# action은 스칼라 → (1,)로 포장
		actions = np.array([int(action)], dtype=np.int64)
		obs, reward, terminated, truncated, info = self.core.step(actions)
		return (
			obs[0],
			float(reward[0]),
			bool(terminated[0]),
			bool(truncated[0]),
			self._slice_info(info, 0),
		)

	def render(self):
		# 텍스트 보드 렌더링
		row16 = self.core.get_original_boards()  # (N, 4) uint16
		return self.core.render_obs(row16)

	def _slice_info(self, info: dict, i: int) -> dict:
		# core의 info는 배치 형태의 항목을 가질 수 있음 → 단일 env로 잘라서 반환
		out = {}
		for k, v in info.items():
			if isinstance(v, np.ndarray) and v.shape[0] == 1:
				out[k] = v[0]
			else:
				out[k] = v
		return out


# =========================
# 2) SB3 VecEnv 어댑터
# =========================

class SB3Batch2048VecEnv(VecEnv):
	"""
	Batch2048Core → SB3 호환 VecEnv 어댑터.
	- core.num_envs 개의 환경을 내부적으로 이미 벡터화하여 처리.
	- SB3가 요구하는 step_async/step_wait, 자동 부분 리셋, infos 리스트화 등을 수행.
	- observation_space / action_space는 '단일 환경 기준'으로 노출.
	"""
	def __init__(self, core):
		# core는 Batch2048Core 인스턴스
		self.core = core
		self.num_envs = int(core.num_envs)

		# SB3의 VecEnv 계약: 단일 환경 기준 space를 노출
		self.observation_space = _single_obs_space_for(core.obs_mode)
		self.action_space = spaces.Discrete(4)

		# VecEnv 내부 상태
		self._waiting = False
		self._actions = None

	@classmethod
	def from_params(cls, *, obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=1024, seed=None, p4=0.1):
		core = Batch2048Core(obs_mode=obs_mode, num_envs=num_envs, seed=seed, p4=p4)
		return cls(core)

	def reset(self):
		obs, info = self.core.reset()
		# SB3는 (num_envs, *obs_shape) 배열과 infos(list[dict])를 기대
		# infos = self._split_info(info)
		return obs

	def step_async(self, actions):
		"""
		SB3가 넘겨주는 actions: shape=(num_envs,), dtype=int
		"""
		if isinstance(actions, np.ndarray):
			if actions.shape != (self.num_envs,):
				raise ValueError(f"actions must have shape ({self.num_envs},), got {actions.shape}")
			self._actions = actions.astype(np.int64, copy=False)
		else:
			_arr = np.asarray(actions, dtype=np.int64)
			if _arr.shape != (self.num_envs,):
				raise ValueError(f"actions must have shape ({self.num_envs},), got {_arr.shape}")
			self._actions = _arr
		self._waiting = True

	def step_wait(self):
		assert self._waiting, "step_wait() called without step_async()"
		self._waiting = False

		obs_step, reward, terminated, truncated, info_step = self.core.step(self._actions)
		done = (terminated | truncated)

		if np.any(terminated):
			obs_before = obs_step.copy()
			obs_after, info_reset = self.core.reset(mask=terminated)
			infos = self._split_info(info_step)
			infos_reset = self._split_info(info_reset)
			for i in range(self.num_envs):
				# reset 관련 키(legal_actions/reset_mask 등) 추가
				infos[i].update(infos_reset[i])
			# terminal_observation 채우기(terminated인 env만)
			for i in np.nonzero(terminated)[0]:
				infos[i] = dict(infos[i])  # copy
				infos[i]["terminal_observation"] = obs_before[i]
			# 리턴할 obs는 리셋 반영된 최신 상태
			obs = obs_after
		else:
			obs = obs_step
			infos = self._split_info(info_step)

		return obs, reward, done, infos

	def close(self):
		pass

	# ---- VecEnv 필수 보조 메서드 ----

	def get_attr(self, name, indices=None):
		# 1) VecEnv 자신에게 있는 속성/메서드 우선
		if hasattr(self, name):
			return [getattr(self, name)]
		# 2) 그 다음 core에서 찾기
		if hasattr(self.core, name):
			return [getattr(self.core, name)]
		raise AttributeError(f"{name!r} not found on VecEnv or core")

	def set_attr(self, name, values, indices=None):
		setattr(self.core, name, values)

	def env_method(self, method_name, *args, indices=None, **kwargs):
		if hasattr(self, method_name):
			return [getattr(self, method_name)(*args, **kwargs)]
		if hasattr(self.core, method_name):
			return [getattr(self.core, method_name)(*args, **kwargs)]
		raise AttributeError(f"Method {method_name!r} not found on VecEnv or core")

	def env_is_wrapped(self, wrapper_class, indices=None):
		return [False]

	def action_masks(self) -> np.ndarray:
		# shape = (num_envs, n_actions) = (N, 4)
		# True = 유효, False = 불가
		return self.core._last_legal_flags.astype(np.bool_, copy=False)

	# ---- 내부 유틸 ----

	def _split_info(self, info_batch: dict) -> list[dict]:
		"""
		core의 info(dict of batch-arrays)를 SB3가 기대하는 list[dict]로 변환.
		- 길이 num_envs의 리스트를 만들고, 각 키별로 i번째 항목만 슬라이스해서 dict 구성.
		"""
		infos = [dict() for _ in range(self.num_envs)]
		for k, v in info_batch.items():
			# v가 배치 배열이면 env별로 슬라이스, 아니면 그대로 복제
			if isinstance(v, np.ndarray) and v.shape[0] == self.num_envs:
				for i in range(self.num_envs):
					infos[i][k] = v[i]
			else:
				for i in range(self.num_envs):
					infos[i][k] = v
		return infos
