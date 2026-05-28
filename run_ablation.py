# -*- coding: utf-8 -*-
"""
run_ablation.py — Robust ablation orchestrator (supports hyphen-leading set names)

Why you saw "unrecognized arguments: -no_gbias ...":
- Argparse treats tokens starting with '-' as new flags, not values to --sets.
This version fixes it by providing a safer interface:

You can now specify ablation sets in ANY of these ways:
1) CSV string (RECOMMENDED):
   --sets_csv "baseline,-no_gbias,-no_mbias,-no_innorm,-no_dyn,-no_cos,-no_lagnorm,pool_meanstd,+stem,+tv,+smooth,+diff"

2) After a double-dash to stop parsing:
   --sets -- -no_gbias -no_mbias +diff baseline

3) As plain values to --sets but ONLY non-hyphen ones (e.g., 'baseline' or '+diff').

It still shells out to your main script with a {feature}/{out}/{seed}/{lag}/{extra} template.
"""
import os, json, argparse, subprocess, shlex, csv, sys
from pathlib import Path

DEFAULT_SETS = {
    "baseline":      "",
    "-no_gbias":     "--no_gaussian_bias",
    "-no_mbias":     "--no_motion_bias",
    "-no_innorm":    "--no_input_norm",
    "-no_dyn":       "--no_dynamics",
    "-no_cos":       "--no_cos",
    "-no_lagnorm":   "--no_lag_norm",
    "pool_meanstd":  "--pooling mean_std",
    "+stem":         "--use_temporal_stem",
    "+tv":           "--lag_tv_lambda 0.01",
    "+smooth":       "--label_smoothing 0.05",
    "+diff":         "--use_diffusion --diffusion_lambda 0.1",
}

def run_once(cmd, log_path: Path, mapping, set_name):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Store traceability files
    (log_path.parent / "ablation_cmd.txt").write_text(cmd, encoding="utf-8")
    (log_path.parent / "sets_map_used.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    with open(log_path, "w") as lf:
        print("[RUN]", cmd)
        p = subprocess.run(shlex.split(cmd), stdout=lf, stderr=subprocess.STDOUT)
        return p.returncode

def read_summary(run_dir: Path):
    sj = run_dir / "summary.json"
    if sj.exists():
        try:
            return json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            pass
    folds = []
    for d in sorted(run_dir.glob("fold_*")):
        mj = d / "metrics.json"
        if mj.exists():
            try:
                m = json.loads(mj.read_text(encoding="utf-8"))
                m["fold_dir"] = str(d)
                folds.append(m)
            except Exception:
                continue
    if not folds:
        return None
    accs = [f.get("test_acc", float("nan")) for f in folds]
    f1s  = [f.get("test_f1", float("nan")) for f in folds]
    return {
        "folds": folds,
        "mean_test_acc": sum(accs)/len(accs) if len(accs)>0 else float("nan"),
        "mean_test_f1":  sum(f1s)/len(f1s)   if len(f1s)>0  else float("nan"),
    }

DEFAULT_SETS = {
    "baseline": "",
    "-laca": "--align_mode direct",
    "+laca": "--align_mode lagaware",
    "pool_avg": "--pooling mean",
    "pool_max": "--pooling max",
    "pool_attn": "--pooling attn_mean_max",
    "pool_attn_std": "--pooling attn_mean_max_std",
    "attn_only": "--pooling attn",
    "-diff": "--use_diff 0",
    "+diff": "--use_diff 1",
    "-gbias": "--use_gaussian_bias 0",
    "+gbias": "--use_gaussian_bias 1",
    "-dyn": "--use_dynamics 0",
    "+dyn": "--use_dynamics 1",
    "-cos": "--use_cos 0",
    "+cos": "--use_cos 1",
    "-lagnorm": "--use_lag_norm 0",
    "+lagnorm": "--use_lag_norm 1"
}

def parse_args_robust():
    ap = argparse.ArgumentParser("Ablation Orchestrator")
    ap.add_argument("--main_cmd", type=str, required=True,
                    help="Command template with placeholders: {feature} {out} {seed} {lag} {extra}")
    ap.add_argument("--out_root", type=str, default="./runs_ablation")
    ap.add_argument("--features", nargs="+", default=["skeleton"])
    ap.add_argument("--seeds", nargs="+", default=["42"])
    ap.add_argument("--lag", type=int, default=12)

    # Robust set selection
    ap.add_argument("--sets_csv", type=str, default=None,
                    help="Comma-separated list of ablation set names (handles hyphen-leading safely).")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="Optional list of set names (avoid values starting with '-' unless you use '--').")
    ap.add_argument("--dry", action="store_true", help="Print commands only")

    # Use parse_known_args to capture anything after '--' as raw set names
    args, remainder = ap.parse_known_args()

    # 解析 sets（消融集合名），获取要运行的设置
    sets = None
    if args.sets_csv:
        sets = [s.strip() for s in args.sets_csv.split(",") if s.strip()]
    elif args.sets:
        sets = args.sets
    elif remainder:
        # 处理命令行中的剩余参数
        sets = [tok for tok in remainder if tok.strip()]
    else:
        sets = list(DEFAULT_SETS.keys())

    mapping = DEFAULT_SETS

    return args, sets, mapping

def main():
    args, sets, mapping = parse_args_robust()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Ablation sets:", sets)
    print("[INFO] Using mapping for sets -> flags:")
    for k in sets:
        print(f"  - {k}: '{mapping.get(k, k)}'")

    results = []
    for feat in args.features:
        for seed in args.seeds:
            for set_name in sets:
                # 获取 extra 参数
                extra = mapping.get(set_name, set_name)  # fallback: raw flags
                tag = f"{feat}_{set_name}_seed{seed}"
                run_dir = out_root / tag
                out_dir = str(run_dir)
                # 确保将 extra 插入到命令中
                cmd = args.main_cmd.format(feature=feat, out=out_dir, seed=seed, lag=args.lag, extra=extra)
                log_path = run_dir / "run.log"
                if args.dry:
                    print("[DRY]", cmd); 
                    # 预运行时保存命令并退出
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "ablation_cmd.txt").write_text(cmd, encoding="utf-8")
                    (run_dir / "sets_map_used.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
                    continue
                rc = run_once(cmd, log_path, mapping, set_name)
                summ = read_summary(run_dir) or {}
                row = {"tag": tag, "returncode": rc}
                row.update({k:v for k,v in summ.items() if not isinstance(v, list)})
                results.append(row)

    # Save summary CSV & JSON
    csv_path = out_root / "ablation_summary.csv"
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in results: w.writerow(r)
    (out_root / "ablation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Saved: {csv_path} and ablation_summary.json")

if __name__ == "__main__":
    main()
