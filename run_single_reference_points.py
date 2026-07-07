"""
Run four single-reference-point Bayesian CPT estimations and compare them.

Each estimation fixes the reference-point weights to one canonical component:
SQ, PA, LE, or FE. Results are written to separate folders, and a comparison
table is saved under the output root.

Examples:
    python3 run_single_reference_points.py
    python3 run_single_reference_points.py --draws 500 --chains 2 --clusters 1
    python3 run_single_reference_points.py --rerun --cores 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

REFERENCE_POINTS = (
    ("SQ", "status quo"),
    ("PA", "partial adaptation"),
    ("LE", "lagged expectation"),
    ("FE", "forward expectation"),
)

OPTION_ENV = {
    "method": "EB_GLOBAL_METHOD",
    "utility": "EB_GLOBAL_UTILITY",
    "discounting": "EB_GLOBAL_DISCOUNTING",
    "clusters": "EB_GLOBAL_CLUSTER",
    "draws": "EB_GLOBAL_DRAWS",
    "chains": "EB_GLOBAL_CHAINS",
    "seed": "EB_GLOBAL_SEED",
    "ksi_mode": "EB_GLOBAL_KSI_MODE",
    "delta_bounds": "EB_GLOBAL_DELTA_BOUNDS",
    "data": "EB_GLOBAL_DATA",
    "lottery_set": "EB_GLOBAL_LOTTERY_SET",
    "lottery_ids": "EB_GLOBAL_LOTTERY_IDS",
}

THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _set_pytensor_flag(flags, key, value):
    parts = [
        part.strip()
        for part in flags.split(",")
        if part.strip() and not part.strip().startswith(f"{key}=")
    ]
    parts.append(f"{key}={value}")
    return ",".join(parts)


def _env_value(value):
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _finite_sample_stat_values(data_array):
    values = data_array.values

    def _iter_numeric(obj):
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            for item in obj.flat:
                yield from _iter_numeric(item)
            return
        arr = np.asarray(obj, dtype=float).ravel()
        for value in arr:
            if np.isfinite(value):
                yield float(value)

    return list(_iter_numeric(values))


def _is_completed(output_dir):
    has_expected_outputs = (
        (output_dir / "summary.csv").exists()
        and (output_dir / "summary_reference_weights.csv").exists()
        and (output_dir / "summary_model_fit.csv").exists()
        and (output_dir / "summary_diagnostics.csv").exists()
    )
    status_path = output_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            if status.get("status") == "completed" and has_expected_outputs:
                return True
        except json.JSONDecodeError:
            pass
    return has_expected_outputs


def _job_env(args, reference_point, output_dir):
    env = os.environ.copy()
    for attr, env_key in OPTION_ENV.items():
        value = getattr(args, attr)
        if value not in (None, ""):
            env[env_key] = _env_value(value)

    env["EB_GLOBAL_FIXED_REFERENCE_POINT"] = reference_point
    env["EB_GLOBAL_OUTPUT_DIR"] = str(output_dir)
    env["EB_GLOBAL_PROGRESSBAR"] = "1" if args.progress else "0"
    if args.cores is not None:
        env["EB_GLOBAL_SMC_CORES"] = str(args.cores)

    for key, value in THREAD_LIMITS.items():
        env[key] = value

    env["MPLCONFIGDIR"] = str(output_dir / "_matplotlib")
    env["PYTENSOR_FLAGS"] = _set_pytensor_flag(
        env.get("PYTENSOR_FLAGS", ""),
        "base_compiledir",
        str(output_dir / "_pytensor"),
    )
    return env


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_batch_defaults(path="BatchSettings.py"):
    settings_path = Path(path)
    if not settings_path.is_absolute():
        settings_path = ROOT / settings_path
    spec = importlib.util.spec_from_file_location("single_reference_batch_settings", settings_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "DEFAULTS", {}))


def _apply_batch_defaults(args, defaults):
    option_map = {
        "method": "method",
        "utility": "utility",
        "discounting": "discounting",
        "clusters": "clusters",
        "draws": "draws",
        "chains": "chains",
        "cores": "cores",
        "seed": "seed",
        "ksi_mode": "ksi_mode",
        "delta_bounds": "delta_bounds",
        "data": "data",
        "lottery_set": "lottery_set",
        "lottery_ids": "lottery_ids",
    }
    for attr, key in option_map.items():
        if getattr(args, attr, None) in (None, "") and key in defaults:
            setattr(args, attr, defaults[key])
    if not args.progress and defaults.get("progressbar") is True:
        args.progress = True
    return args


def _run_one(args, reference_point, description, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_matplotlib").mkdir(exist_ok=True)
    (output_dir / "_pytensor").mkdir(exist_ok=True)

    if _is_completed(output_dir) and not args.rerun:
        print(f"[skip] {reference_point} already completed -> {output_dir}")
        return 0

    settings_payload = {
        "reference_point": reference_point,
        "description": description,
        "defaults_source": "BatchSettings.DEFAULTS",
        "output_dir": str(output_dir),
        "draws": args.draws,
        "chains": args.chains,
        "clusters": args.clusters,
        "seed": args.seed,
        "cores": args.cores,
        "method": args.method,
        "utility": args.utility,
        "discounting": args.discounting,
        "lottery_set": args.lottery_set,
        "lottery_ids": args.lottery_ids,
        "ksi_mode": args.ksi_mode,
        "delta_bounds": args.delta_bounds,
    }
    _write_json(output_dir / "settings.json", settings_payload)
    _write_json(
        output_dir / "status.json",
        {
            "reference_point": reference_point,
            "status": "running",
            "started_at": _now(),
        },
    )

    command = [args.python, "main.py"]
    if args.load:
        command.append("--load")

    print(f"[start] {reference_point:<2} {description:<20} -> {output_dir}")
    started_at = time.time()
    log_path = output_dir / "run.log"
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(f"\n=== START {_now()} reference_point={reference_point} ===\n")
        log_file.write(f"command: {' '.join(command)}\n")
        log_file.write(f"output_dir: {output_dir}\n\n")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_job_env(args, reference_point, output_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.write(f"\n=== END {_now()} returncode={completed.returncode} ===\n")

    elapsed = round(time.time() - started_at, 1)
    status = "completed" if completed.returncode == 0 else "failed"
    _write_json(
        output_dir / "status.json",
        {
            "reference_point": reference_point,
            "status": status,
            "returncode": completed.returncode,
            "started_at": datetime.fromtimestamp(started_at).astimezone().isoformat(timespec="seconds"),
            "finished_at": _now(),
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
            "log_path": str(log_path),
        },
    )
    print(f"[{status}] {reference_point:<2} rc={completed.returncode} elapsed={elapsed}s")
    return completed.returncode


def _read_log_marginal_likelihood(output_dir):
    fit_path = output_dir / "summary_model_fit.csv"
    if fit_path.exists():
        try:
            fit_df = pd.read_csv(fit_path)
            row = fit_df[fit_df["metric"] == "log_marginal_likelihood"]
            if not row.empty:
                value = pd.to_numeric(row.iloc[0]["value"], errors="coerce")
                if pd.notna(value):
                    return float(value), "summary_model_fit"
        except Exception as exc:
            print(f"[warn] Could not read log marginal likelihood from {fit_path}: {exc}")

    trace_path = output_dir / "trace.nc"
    if not trace_path.exists():
        values = []
    else:
        try:
            import arviz as az

            idata = az.from_netcdf(trace_path)
            if not hasattr(idata, "sample_stats"):
                values = []
            elif "log_marginal_likelihood" not in idata.sample_stats:
                values = []
            else:
                values = _finite_sample_stat_values(idata.sample_stats["log_marginal_likelihood"])
            if values:
                return float(np.mean(values)), "trace"
        except Exception as exc:
            print(f"[warn] Could not read log marginal likelihood from {trace_path}: {exc}")

    log_path = output_dir / "run.log"
    if log_path.exists():
        matches = re.findall(
            r"Log marginal likelihood:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            log_path.read_text(),
        )
        if matches:
            return float(matches[-1]), "run_log"
    return np.nan, "missing"


def _read_status(output_dir):
    completed_by_outputs = (
        (output_dir / "summary.csv").exists()
        and (output_dir / "summary_reference_weights.csv").exists()
        and (output_dir / "summary_model_fit.csv").exists()
        and (output_dir / "summary_diagnostics.csv").exists()
    )
    status_path = output_dir / "status.json"
    if not status_path.exists():
        return "completed" if completed_by_outputs else "missing"
    try:
        status = json.loads(status_path.read_text()).get("status", "unknown")
        if status == "completed" and not completed_by_outputs:
            return "incomplete"
        if status != "completed" and completed_by_outputs:
            return "completed"
        return status
    except json.JSONDecodeError:
        return "completed" if completed_by_outputs else "unknown"


def _collect_results(output_root, jobs):
    rows = []
    cluster_frames = []
    reference_frames = []
    fit_frames = []

    for reference_point, description, output_dir in jobs:
        summary_path = output_dir / "summary.csv"
        ref_path = output_dir / "summary_reference_weights.csv"
        fit_path = output_dir / "summary_model_fit.csv"
        log_ml, log_ml_source = _read_log_marginal_likelihood(output_dir)

        rows.append(
            {
                "reference_point": reference_point,
                "description": description,
                "status": _read_status(output_dir),
                "log_marginal_likelihood": log_ml,
                "log_marginal_likelihood_source": log_ml_source,
                "output_dir": str(output_dir),
            }
        )

        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, "reference_point", reference_point)
            cluster_frames.append(df)

        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            ref_df.insert(0, "reference_point", reference_point)
            reference_frames.append(ref_df)

        if fit_path.exists():
            fit_df = pd.read_csv(fit_path)
            fit_df.insert(0, "reference_point", reference_point)
            fit_frames.append(fit_df)

    comparison = pd.DataFrame(rows)
    finite = comparison["log_marginal_likelihood"].replace([np.inf, -np.inf], np.nan).notna()
    comparison["rank_by_log_ml"] = np.nan
    comparison["delta_log_ml"] = np.nan
    comparison["model_probability_equal_prior"] = np.nan

    if finite.any():
        log_ml = comparison.loc[finite, "log_marginal_likelihood"].astype(float)
        best = float(log_ml.max())
        log_norm = best + np.log(np.exp(log_ml - best).sum())
        comparison.loc[finite, "delta_log_ml"] = log_ml - best
        comparison.loc[finite, "model_probability_equal_prior"] = np.exp(log_ml - log_norm)
        comparison.loc[finite, "rank_by_log_ml"] = (
            comparison.loc[finite, "log_marginal_likelihood"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    comparison = comparison.sort_values(
        by=["rank_by_log_ml", "reference_point"],
        na_position="last",
    )
    comparison_path = output_root / "single_reference_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    if cluster_frames:
        cluster_path = output_root / "single_reference_cluster_params.csv"
        pd.concat(cluster_frames, ignore_index=True).to_csv(cluster_path, index=False)

    if reference_frames:
        reference_path = output_root / "single_reference_fixed_weights.csv"
        pd.concat(reference_frames, ignore_index=True).to_csv(reference_path, index=False)

    if fit_frames:
        fit_path = output_root / "single_reference_model_fit.csv"
        pd.concat(fit_frames, ignore_index=True).to_csv(fit_path, index=False)

    return comparison, comparison_path


def _print_comparison(comparison, comparison_path):
    display = comparison[
        [
            "reference_point",
            "status",
            "rank_by_log_ml",
            "log_marginal_likelihood",
            "log_marginal_likelihood_source",
            "delta_log_ml",
            "model_probability_equal_prior",
        ]
    ].copy()
    for col in ("log_marginal_likelihood", "delta_log_ml", "model_probability_equal_prior"):
        display[col] = display[col].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
        )
    display["rank_by_log_ml"] = display["rank_by_log_ml"].map(
        lambda value: "" if pd.isna(value) else str(int(value))
    )

    print("\nSingle-reference comparison")
    print(display.to_string(index=False))
    print(f"\nSaved comparison -> {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SQ/PA/LE/FE single-reference-point estimations and compare them."
    )
    parser.add_argument("--output-root", default="single_reference_outputs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None, help="SMC cores per estimation")
    parser.add_argument("--method", default=None)
    parser.add_argument("--utility", default=None, help="utility function: cara or power")
    parser.add_argument("--discounting", default=None)
    parser.add_argument("--lottery-set", dest="lottery_set", default=None)
    parser.add_argument("--lottery-ids", dest="lottery_ids", default=None)
    parser.add_argument("--ksi-mode", dest="ksi_mode", default=None)
    parser.add_argument("--delta-bounds", dest="delta_bounds", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--rerun", action="store_true", help="rerun even if outputs exist")
    parser.add_argument("--load", action="store_true", help="pass --load to main.py")
    parser.add_argument("--compare-only", action="store_true", help="skip estimation and rebuild comparison files")
    parser.add_argument("--progress", action="store_true", help="enable PyMC progress bars in run logs")
    args = parser.parse_args()
    batch_defaults = _load_batch_defaults()
    args = _apply_batch_defaults(args, batch_defaults)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    utility_label = str(args.utility or os.environ.get("EB_GLOBAL_UTILITY", "cara"))
    utility_label = utility_label.strip().lower().replace("-", "_")
    if utility_label not in {"", "cara"}:
        output_root = output_root / f"utility-{utility_label}"
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (reference_point, description, output_root / f"rp-{reference_point}")
        for reference_point, description in REFERENCE_POINTS
    ]

    returncodes = []
    if not args.compare_only:
        for reference_point, description, output_dir in jobs:
            returncodes.append(_run_one(args, reference_point, description, output_dir))

    comparison, comparison_path = _collect_results(output_root, jobs)
    _print_comparison(comparison, comparison_path)

    if returncodes and any(code != 0 for code in returncodes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
