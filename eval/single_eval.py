from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from env.batch2048_core import Batch2048Core


ROOT_DIR = Path(__file__).parent.parent

core = Batch2048Core(num_envs=5, obs_mode=Batch2048Core.ObsMode.ONEHOT256)

# 모델 불러오기
model = MaskablePPO.load(ROOT_DIR / "tb_2048/PPO_20/checkpoints/latest_model_final.zip")


obs, infos = core.reset()
alive = np.ones(core.num_envs, dtype=bool)
scores = np.zeros(core.num_envs, dtype=np.float64)
step = 0

while alive.any():
	# 행동 마스크 (불법 이동 방지)
	action_masks = infos["legal_actions"]
	action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

	# 스텝 진행
	obs, reward, terminated, truncated, infos = core.step(action)
	scores += reward
	step += 1
	dones = terminated | truncated
	alive &= ~dones

	# 보드 상태 출력 (core에서 직접 접근)
	print(f"\nStep {step}, Action {action}, Reward {reward}")
	print(core.render_obs(obs))

print("\n===== All Env Game Over =====")
print(f"Final Score: {scores}")
print(f"Max Tile: {1 << core.best_tile().astype(np.int32)}")


