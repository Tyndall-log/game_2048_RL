# train_utils.py
from __future__ import annotations
from pathlib import Path
from collections import deque
from typing import Optional
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MonitorAndTimedCheckpointCallback(BaseCallback):
	"""
	VecMonitor의 episode 요약을 이용해 최근 n개 에피소드 누적보상 평균을 텐서보드에 기록하고,
	일정 시간 간격마다 최신 체크포인트로 모델을 덮어씁니다.

	- 기록되는 보상 스케일:
		- VecMonitor가 VecNormalize '안쪽'이면 원보상(raw)
		- VecMonitor가 VecNormalize '바깥쪽'이면 정규화 보상(norm)

	Args:
		rolling_n (int): 최근 n개 에피소드 리턴을 평균 내어 기록.
		rolling_tag (str): 텐서보드에 기록할 태그 이름.
		save_interval_sec (Optional[int]): 체크포인트 저장 간격(초). None이면 저장 비활성화.
		save_subdir (str): logger.dir 하위 저장 폴더 이름.
		save_basename (str): 저장 파일 기본 이름(확장자는 SB3가 .zip으로 자동 추가).
		save_on_train_end (bool): 학습 종료 시 마지막으로 한 번 저장할지 여부.
		verbose (int): 출력 수준.
	"""

	def __init__(
		self,
		rolling_n: int = 128,
		rolling_tag: str = "metrics/rolling_ep_return_from_monitor",
		save_interval_sec: Optional[int] = None,
		save_subdir: str = "checkpoints",
		save_basename: str = "latest_model",
		save_on_train_end: bool = True,
		verbose: int = 0,
	):
		super().__init__(verbose)
		# logging
		self.rolling_n = int(rolling_n)
		self.rolling_tag = str(rolling_tag)
		self._recent: Optional[deque] = None

		# checkpoint
		self.save_interval_sec = int(save_interval_sec) if save_interval_sec is not None else None
		self.save_subdir = str(save_subdir)
		self.save_basename = str(save_basename)
		self.save_on_train_end = bool(save_on_train_end)
		self._save_dir: Optional[str] = None
		self._save_path: Optional[str] = None
		self._last_save_time: Optional[float] = None

	# ----------------------- SB3 lifecycle hooks -----------------------

	def _on_training_start(self) -> None:
		self._recent = deque(maxlen=self.rolling_n)

		if self.save_interval_sec is not None:
			self._save_dir = Path(self.logger.dir) / self.save_subdir
			self._save_dir.mkdir(parents=True, exist_ok=True)
			self._save_path = self._save_dir / self.save_basename
			self._last_save_time = time.time()
			if self.verbose:
				print(f"[Callback] Checkpoints -> {self._save_path}.zip (every {self.save_interval_sec}s)")

	def _on_step(self) -> bool:
		# --------- 1) Rolling episode return from VecMonitor ---------
		infos = self.locals.get("infos")   # list[dict], len = num_envs
		dones = self.locals.get("dones")   # np.ndarray(bool) or list
		if infos is not None and dones is not None:
			dones = np.asarray(dones)
			if np.any(dones):
				finished_idx = np.nonzero(dones)[0]
				for i in finished_idx:
					ep = infos[i].get("episode", None)
					if ep is None:
						continue
					# ep: {"r": return, "l": length, "t": seconds}
					ep_r = float(ep.get("r", 0.0))
					self._recent.append(ep_r)

				if self._recent and len(self._recent) > 0:
					mean_ep_r = float(np.mean(self._recent))
					self.logger.record(self.rolling_tag, mean_ep_r)

		# --------- 2) Time-based latest checkpoint overwrite ---------
		if self.save_interval_sec is not None and self._save_path is not None:
			now = time.time()
			if (self._last_save_time is None) or (now - self._last_save_time >= self.save_interval_sec):
				self._save_latest()
				self._last_save_time = now

		return True

	def _on_training_end(self) -> None:
		if self.save_on_train_end and self.save_interval_sec is not None and self._save_path is not None:
			self._save_latest(suffix="_final")
			if self.verbose:
				print(f"[Callback] Final checkpoint saved at end of training.")

	# ----------------------- helpers -----------------------

	def _save_latest(self, suffix: str = "") -> None:
		"""
		최신 체크포인트를 같은 파일명으로 덮어쓰기.
		suffix를 주면 파일명 뒤에 덧붙여 저장(예: latest_model_final.zip)
		"""
		base: Path = self._save_path if suffix == "" else Path(str(self._save_path) + suffix)
		self.model.save(str(base))  # SB3 save는 str path 필요
		if self.verbose:
			print(f"[Checkpoint] Saved -> {base}.zip @ step={self.n_calls}, timesteps={self.num_timesteps}")