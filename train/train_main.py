# train_main.py
from __future__ import annotations
from pathlib import Path
from torch import nn
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from sb3_contrib import MaskablePPO

from env.batch2048_core import Batch2048Core
from env.adapters_2048 import SB3Batch2048VecEnv
from train.train_utils import MonitorAndTimedCheckpointCallback

ROOT_DIR = Path(__file__).parent.parent
LOG_DIR = ROOT_DIR / "tb_2048"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 1) 코어 + VecEnv
core = Batch2048Core(obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=2**13, seed=42)
venv = SB3Batch2048VecEnv(core)

# 2) (중요) 모니터는 원보상을 보고 싶으면 VecNormalize '안쪽'에 둔다
venv = VecMonitor(venv)  # 에피소드 리턴/길이 로그 (원보상 기준)

# 3) 보상 정규화 (관측은 onehot이니 norm_obs=False)
venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

# 4) 모델
model = MaskablePPO(
	"MlpPolicy",
	venv,
	policy_kwargs= {
		"activation_fn": nn.ReLU,
		"net_arch": {"pi": [512, 256, 32], "vf": [512, 256, 32]},
	},
	n_steps=2**5,
	batch_size=2048,
	n_epochs=4,
	learning_rate=3e-4,
	gamma=0.99,
	gae_lambda=0.95,
	clip_range=0.2,
	vf_coef=0.5,
	ent_coef=0.0,
	max_grad_norm=0.5,
	tensorboard_log=str(LOG_DIR),
	verbose=1,
	device="auto",
)

# 5) 콜백 (최근 512개 에피소드의 평균을 기록)
rolling_cb = MonitorAndTimedCheckpointCallback(
	rolling_n=512,
	save_interval_sec=300,  # 5분마다 저장
	save_subdir="checkpoints",
	save_basename="latest_model",
	save_on_train_end=True,
	verbose=1,
)

# 6) 학습
TOTAL_STEPS = 10_000_000_000
model.learn(total_timesteps=TOTAL_STEPS, callback=rolling_cb)

# # 7) 저장 (VecNormalize 파라미터 포함)
# name= f"{core.obs_mode.name}_256_256_128_relu"
# model.save(f"ppo_{name}.zip")
# venv.save(f"vecnorm_{name}.pkl")

# 실시간 로그 값 미리보고 콘솔 명령어:
# tensorboard --logdir .