import numpy as np
from sb3_contrib import MaskablePPO

from env.batch2048_core import Batch2048Core
from env.adapters_2048 import SB3Batch2048VecEnv

# -----------------------
# 환경 생성
# -----------------------
core = Batch2048Core(num_envs=1, obs_mode=Batch2048Core.ObsMode.UINT8x16)
venv = SB3Batch2048VecEnv(core)

# 모델 불러오기
model = MaskablePPO.load("ppo_2048_masked_8x16_256_256_relu.zip")

# -----------------------
# 1회 플레이
# -----------------------
obs = venv.reset()
done = False
score = 0
step = 0


def print_board(obs: np.ndarray):
	# obs: (N, 4, 4) uint8
	for row in obs.reshape(-1, 4, 4).swapaxes(0, 1):
		for r in row:
			print(" ".join(f"{(1 << int(v)) if v > 0 else 0:4d}" for v in r), end="   |   ")
		print()

while not done:
	# 행동 마스크 (불법 이동 방지)
	action_masks = venv.get_attr("action_masks")[0]()  # single env
	action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

	# 스텝 진행
	obs, reward, dones, infos = venv.step(action)
	score += reward[0]
	step += 1
	done = dones[0]

	# 보드 상태 출력 (core에서 직접 접근)
	print(f"\nStep {step}, Action {action}, Reward {reward[0]}")
	if done:
		board = infos[0].get("terminal_observation", None)
		if core.obs_mode == "onehot256" and board is not None:
			board = np.argmax(board.reshape(-1, 16), axis=1).reshape(-1,4)
	else:
		board = core.boards_as_uint8x16()
	print_board(board)
	# print(core.render_obs(board))

	if done:
		print("\n===== Game Over =====")
		print(f"Final Score: {score}")
		print(f"Max Tile: {1 << int(board.max())}")
		done = False
		score = 0
		if (1 << int(board.max())) <= 1024:
			done = True
			score = 0

