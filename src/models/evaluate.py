"""Lightweight evaluator: summarise existing metrics JSON files.

This script reads `models/metrics/*.json`, extracts test metrics for
LightGBM, XGBoost and Prophet (when present), prints a short verdict per
model and an optional consolidated summary file.

Usage:
  python -m src.models.evaluate --all [--save-all]
"""

from pathlib import Path
import json
import argparse
from typing import Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = PROJECT_ROOT / "models" / "metrics"


def _verdict_from_r2(r2: float) -> str:
	if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
		return "No R2 available"
	if r2 >= 0.85:
		return "Excellent model -- strong predictive power."
	elif r2 >= 0.7:
		return "Good model -- explains significant variance."
	elif r2 >= 0.5:
		return "Moderate model -- useful but improvable."
	elif r2 >= 0.3:
		return "Weak model -- limited signal."
	elif r2 >= 0:
		return "Poor model -- barely better than mean."
	else:
		return "Model worse than mean -- not useful."


def _score_models(results_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
	tags = list(results_map.keys())
	n = len(tags)
	if n == 0:
		return {}

	maes = {t: float(results_map[t]['test'].get('MAE', results_map[t]['test'].get('mae', float('inf')))) for t in tags}
	rmses = {t: float(results_map[t]['test'].get('RMSE', results_map[t]['test'].get('rmse', float('inf')))) for t in tags}
	r2s = {t: float(results_map[t]['test'].get('R2', results_map[t]['test'].get('r2', float('-inf')))) for t in tags}

	def rank_scores(metric_dict, reverse=False):
		items = sorted(metric_dict.items(), key=lambda kv: kv[1], reverse=reverse)
		scores = {}
		if n == 1:
			scores[items[0][0]] = 1.0
			return scores
		for idx, (tag, _) in enumerate(items):
			scores[tag] = (n - 1 - idx) / (n - 1)
		return scores

	mae_scores = rank_scores(maes, reverse=False)
	rmse_scores = rank_scores(rmses, reverse=False)
	r2_scores = rank_scores(r2s, reverse=True)

	weights = {'MAE': 0.4, 'RMSE': 0.4, 'R2': 0.2}
	final = {}
	for t in tags:
		sc = mae_scores.get(t, 0.0) * weights['MAE'] + rmse_scores.get(t, 0.0) * weights['RMSE'] + r2_scores.get(t, 0.0) * weights['R2']
		final[t] = round(float(sc * 100), 2)
	return final


def evaluate_all(save: bool = False) -> Dict[str, Any]:
	files = sorted(METRICS_PATH.glob('*.json')) if METRICS_PATH.exists() else []
	results_map: Dict[str, Dict[str, Any]] = {}

	for f in files:
		name = f.name.lower()
		if name.startswith('eval_summary'):
			continue

		if any(k in name for k in ('lgbm', 'lgb', 'lightgbm')):
			backend = 'LightGBM'
		elif any(k in name for k in ('xgb', 'xg', 'xgboost')):
			backend = 'XGBoost'
		elif 'prophet' in name:
			backend = 'Prophet'
		else:
			backend = f.stem

		try:
			with open(f, 'r', encoding='utf-8') as fh:
				data = json.load(fh)
		except Exception as e:
			print(f"Failed reading {f}: {e}")
			continue

		# Heuristic: many metric JSONs use keys like MAE_test, RMSE_test, R2_test
		test_metrics = {}
		if isinstance(data, dict):
			# prefer keys that end with _test_corrected, then _test
			for key, val in data.items():
				if key.lower().endswith('_test_corrected'):
					base = key[:-len('_test_corrected')]
					test_metrics[base.upper()] = val
			for key, val in data.items():
				if key.lower().endswith('_test'):
					base = key[:-len('_test')]
					# don't overwrite corrected
					up = base.upper()
					if up not in test_metrics:
						test_metrics[up] = val

			# also accept nested shapes like {'MAE_test':..} under other keys
			if not test_metrics:
				if 'test' in data and isinstance(data['test'], dict):
					for k, v in data['test'].items():
						test_metrics[k.upper()] = v
				elif 'metrics' in data and isinstance(data['metrics'], dict):
					m = data['metrics']
					if 'test' in m and isinstance(m['test'], dict):
						for k, v in m['test'].items():
							test_metrics[k.upper()] = v

		if not test_metrics:
			print(f"No test metrics found in {f.name}, skipping")
			continue

		results_map[backend] = {'test': test_metrics, 'source': str(f)}

		# Pull common metrics
		def _get(metric_key):
			return test_metrics.get(metric_key) or test_metrics.get(metric_key.upper())

		mae = _get('MAE')
		rmse = _get('RMSE')
		r2 = _get('R2')
		try:
			mae_v = float(mae) if mae is not None else float('nan')
		except Exception:
			mae_v = float('nan')
		try:
			rmse_v = float(rmse) if rmse is not None else float('nan')
		except Exception:
			rmse_v = float('nan')
		try:
			r2_v = float(r2) if r2 is not None else float('nan')
		except Exception:
			r2_v = float('nan')

		print(f"\n=== {backend} ({f.name}) ===")
		print(f"Test metrics -> MAE: {mae_v:.3f}, RMSE: {rmse_v:.3f}, R2: {r2_v:.3f}")
		print(f"Verdict: {_verdict_from_r2(r2_v)}\n")

	scores = _score_models(results_map)
	summary = {k: {'metrics': results_map[k]['test'], 'score': scores.get(k)} for k in results_map}

	if save and summary:
		METRICS_PATH.mkdir(parents=True, exist_ok=True)
		import datetime
		ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
		out_path = METRICS_PATH / f'eval_summary_{ts}.json'
		with open(out_path, 'w', encoding='utf-8') as fh:
			json.dump({'summary': summary, 'details': results_map}, fh, indent=2, default=str)
		print(f"Saved consolidated summary to {out_path}")

	if summary:
		print('\nEvaluation summary (score 0-100):')
		for k, v in summary.items():
			met = v.get('metrics')
			sc = v.get('score')
			try:
				maef = float(met.get('MAE', float('nan')))
				rmsef = float(met.get('RMSE', float('nan')))
				r2f = float(met.get('R2', float('nan')))
				print(f"- {k}: score={sc}  MAE={maef:.2f}  RMSE={rmsef:.2f}  R2={r2f:.3f}")
			except Exception:
				print(f"- {k}: score={sc}  metrics={met}")
	else:
		print('No metrics files found or no usable test metrics in metrics files.')

	return {'summary': summary, 'details': results_map}


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description='Summarise existing metrics files')
	parser.add_argument('--all', action='store_true', help='Summarise all metrics files')
	parser.add_argument('--save-all', action='store_true', help='Save consolidated summary')
	args = parser.parse_args(argv)

	if args.all:
		evaluate_all(save=args.save_all)
		return 0
	else:
		print('Run with --all to summarise metrics files in models/metrics')
		return 0


if __name__ == '__main__':
	raise SystemExit(main())

