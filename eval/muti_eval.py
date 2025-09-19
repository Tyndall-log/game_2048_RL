# eval_batch_compare.py
import os
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

from env.batch2048_core import Batch2048Core


class ModelSpec:
	def __init__(self, name: str, path: str, obs_mode: Batch2048Core.ObsMode, color: str):
		self.name = name
		self.path = path
		self.obs_mode = obs_mode
		self.color = color


def evaluate_one_model(model_path: str, model_name: str, obs_mode: Batch2048Core.ObsMode, num_envs: int = 2048, seed: int | None = 123):
	"""
	한 모델을 2048개 환경에서 동시 플레이하고,
	각 env의 '첫 번째 에피소드' 누적 보상을 반환합니다.
	자동 리셋되더라도 alive 마스크로 첫 에피소드만 집계합니다.
	"""
	# --- 벡터 환경 준비 ---
	core = Batch2048Core(num_envs=num_envs, seed=seed, obs_mode=obs_mode)
	obs, infos = core.reset()

	# --- 모델 로드 (env 없이) ---
	model: MaskablePPO = MaskablePPO.load(model_path, device="auto", print_system_info=False)

	# --- 누적 보상/종료 추적 ---
	ep_returns = np.zeros((num_envs,), dtype=np.float64)
	alive = np.ones((num_envs,), dtype=bool)

	# --- 롤아웃: 모든 env가 한 번씩 끝날 때까지 ---
	steps = 0
	t0 = time.time()
	p = tqdm(total=num_envs, desc=f"[{model_name}] Evaluating", unit="ep")
	while alive.any():
		# 진행 상황 표시
		p.update(np.sum(~alive) - p.n)
		# 액션 마스크
		action_masks = infos.get("legal_actions")

		# 정책 호출 (deterministic=True)
		actions, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

		# 스텝
		obs, rewards, terminated, truncated, infos = core.step(actions)
		dones = terminated | truncated  # (num_envs,) bool
		ep_returns += rewards

		# 이번 스텝으로 종료한 env는 alive에서 제외 (그 이후 보상은 무시)
		alive &= ~dones
		steps += 1

	dt = time.time() - t0
	fps = (steps * num_envs) / max(dt, 1e-9)
	# --- 요약 통계 (프린트 대신 반환) ---
	mean_ret = float(ep_returns.mean())
	q25 = float(np.percentile(ep_returns, 25))
	q50 = float(np.median(ep_returns))
	q75 = float(np.percentile(ep_returns, 75))

	# --- 최대 타일 분포 (값 기준: 2^e) ---
	max_e = core.best_tile()  # (num_envs,) uint8, e
	max_val = np.where(max_e > 0, 1 << max_e.astype(np.int32), 0)
	vals, cnts = np.unique(max_val, return_counts=True)
	# 0(빈칸) 제외 및 정렬
	display = [(int(v), int(c)) for v, c in zip(vals, cnts) if v > 0]
	display.sort(key=lambda t: t[0])
	# 백분위/누적 백분위 계산
	total = float(num_envs)
	counts = {v: c for v, c in display}
	percents = {v: (c / total) * 100.0 for v, c in display}
	cum = 0.0
	cum_percents = {}
	for v, c in display:
		cum += (c / total) * 100.0
		cum_percents[v] = cum

	result = {
		"returns": ep_returns,
		"fps": float((steps * num_envs) / max(dt, 1e-9)),
		"time_elapsed": float(dt),
		"steps": int(steps),
		"stats": {"mean": mean_ret, "q25": q25, "median": q50, "q75": q75},
		"max_tile": {
			"counts": counts,
			"percents": percents,      # % 단위
			"cum_percents": cum_percents,  # % 단위 누적
		},
	}
	return result


def evaluate_models(model_specs: list[ModelSpec], num_envs=2048, seed=123):
	"""
	Args:
		model_specs: list of ModelSpec
	Returns:
		results: dict {name: returns(np.array shape [num_envs])}
		palette: dict {name: color_hex}
	"""
	results = {}
	palette = {}
	for spec in model_specs:
		assert os.path.exists(spec.path), f"missing: {spec.path}"
		res = evaluate_one_model(spec.path, spec.name, spec.obs_mode, num_envs=num_envs, seed=seed)
		results[spec.name] = res
		# normalize hex color (allow without '#')
		color_hex = spec.color if spec.color.startswith('#') else f"#{spec.color}"
		palette[spec.name] = color_hex
	return results, palette


def plot_results(results: dict[str, dict], palette: dict[str, str], title_prefix: str = "Batch Eval (N=2048)"):
	"""
	두 플롯 + 1 히트맵:
	  1) 모델별 return 분포 violinplot + swarmplot (수평)
	  2) 모델별 return boxplot (사분위, 수평)
	  3) 모델×최대타일(값) 히트맵: 퍼센트로 컬러맵, 셀에는 "count (pct%)" 표시
	또한 콘솔에 두 가지 표를 출력:
	  - 요약 통계표(평균, 25/50/75%, FPS, time)
	  - 모델별 최대 타일 분포표(값, 개수, 백분위, 누적 백분위)
	"""
	# --- 긴 형태 데이터프레임 (returns만 추출) ---
	rows = []
	for name, res in results.items():
		arr = res["returns"]
		for x in arr:
			rows.append((name, float(x)))
	df = pd.DataFrame(rows, columns=["model", "return"])

	# seaborn 스타일
	sns.set_theme(style="whitegrid", context="talk")

	# --- 콘솔: 요약 통계 테이블 ---
	print("\n[Summary Statistics]")
	sum_rows = []
	for name, res in results.items():
		st = res["stats"]
		sum_rows.append({
			"Model": name,
			"Mean": f"{st['mean']:.1f}",
			"Q25": f"{st['q25']:.1f}",
			"Median": f"{st['median']:.1f}",
			"Q75": f"{st['q75']:.1f}",
			"FPS": f"{res['fps']:,.0f}",
			"Time(s)": f"{res['time_elapsed']:.2f}",
		})
	print(pd.DataFrame(sum_rows).to_string(index=False))

	# 1) violinplot + swarmplot (수평)
	plt.figure(figsize=(16, 9), dpi=300)
	ax = sns.violinplot(
		data=df, x="return", y="model",
		hue="model",  # palette 적용을 위한 hue
		inner=None, palette=palette,
		cut=0, fill=False, linewidth=1.2,
	)
	leg = ax.get_legend()
	if leg is not None:
		leg.remove()
	sns.swarmplot(
		data=df, x="return", y="model",
		hue="model", palette=palette,
		size=1, alpha=1, dodge=False, linewidth=0, zorder=10,
	)
	leg2 = ax.get_legend()
	if leg2 is not None:
		leg2.remove()
	ax.set_title(f"{title_prefix} — violin + swarm")
	ax.set_xlabel("Return")
	ax.set_ylabel("Model")
	plt.tight_layout()
	plt.show()

	# 2) boxplot (5–95% whiskers, 수평)
	plt.figure(figsize=(16, 9), dpi=300)
	ax2 = sns.boxplot(data=df, x="return", y="model", hue="model", whis=(5, 95), palette=palette, orient="h")
	leg = ax2.get_legend()
	if leg is not None:
		leg.remove()
	ax2.set_title(f"{title_prefix} — distribution (5–95% whiskers)")
	ax2.set_xlabel("Return")
	ax2.set_ylabel("Model")
	plt.tight_layout()
	plt.show()

	# --- 최대 타일 히트맵 준비 ---
	# 모든 모델에서 등장한 타일 값의 정렬된 합집합 컬럼
	all_tiles = sorted({v for res in results.values() for v in res["max_tile"]["counts"].keys()})
	models = list(results.keys())

	# 퍼센트 매트릭스와 주석 문자열 준비
	perc_mat = np.zeros((len(models), len(all_tiles)), dtype=float)
	annot = np.empty((len(models), len(all_tiles)), dtype=object)
	for i, m in enumerate(models):
		res = results[m]
		counts = res["max_tile"]["counts"]
		percs = res["max_tile"]["percents"]
		for j, t in enumerate(all_tiles):
			c = int(counts.get(t, 0))
			p = float(percs.get(t, 0.0))
			perc_mat[i, j] = p
			annot[i, j] = f"{c}\n({p:.1f}%)"

	# 3) 히트맵 (퍼센트 컬러, count+% 주석)
	plt.figure(figsize=(max(16, 0.7 * len(all_tiles) + 6), 4 + 1 * len(models)), dpi=300)
	ax3 = sns.heatmap(
		perc_mat, annot=annot, fmt="",
		cmap="Blues", cbar_kws={"label": "% of episodes"},
		xticklabels=[str(t) for t in all_tiles], yticklabels=models,
		linewidths=0.5, linecolor="white", square=False
	)
	ax3.set_title(f"{title_prefix} — Max Tile heatmap (percent; annotations=count)")
	ax3.set_xlabel("Max tile value")
	ax3.set_ylabel("Model")
	plt.tight_layout()
	plt.show()

	# --- 콘솔: 모델별 최대 타일 분포 표 (개수/백분위/누적 백분위) ---
	print("\n[Max Tile Distribution per Model]")
	for m in models:
		res = results[m]
		cnts = res["max_tile"]["counts"]
		percs = res["max_tile"]["percents"]
		cums = res["max_tile"]["cum_percents"]
		tiles = sorted(cnts.keys())
		rows = []
		acc = 0
		for t in tiles:
			c = int(cnts[t])
			p = float(percs[t])
			cp = float(cums[t])
			rows.append({"Tile": t, "Count": c, "Percent": f"{p:.2f}%", "CumPercent": f"{cp:.2f}%"})
		print(f"\nModel: {m}")
		print(pd.DataFrame(rows).to_string(index=False))
	pass


if __name__ == "__main__":
	# 비교할 모델들: (표시이름, 경로, 관측모드, 색상)
	MODELS = [
		ModelSpec(
			name="PPO_7",
			path="../tb_2048/PPO_7/checkpoints/latest_model_final.zip",
			obs_mode=Batch2048Core.ObsMode.ONEHOT256,
			color="e8710a",
		),
		ModelSpec(
			name="PPO_19",
			path="../tb_2048/PPO_19/checkpoints/latest_model_final.zip",
			obs_mode=Batch2048Core.ObsMode.ONEHOT256,
			color="9334e6",
		),
		ModelSpec(
			name="PPO_20",
			path="../tb_2048/PPO_20/checkpoints/latest_model_final.zip",
			obs_mode=Batch2048Core.ObsMode.ONEHOT256,
			color="7cb342",
		),
		ModelSpec(
			name="PPO_21",
			path="../tb_2048/PPO_21/checkpoints/latest_model_final.zip",
			obs_mode=Batch2048Core.ObsMode.ONEHOT256,
			color="425066",
		),
	]

	results, palette = evaluate_models(MODELS, num_envs=2048, seed=42)
	plot_results(results, palette, title_prefix="2048 MaskablePPO")