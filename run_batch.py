"""
Queue and run Bayesian CPT estimations from BatchSettings.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ENV_KEYS = {
    "method": "EB_GLOBAL_METHOD",
    "discounting": "EB_GLOBAL_DISCOUNTING",
    "clusters": "EB_GLOBAL_CLUSTER",
    "draws": "EB_GLOBAL_DRAWS",
    "chains": "EB_GLOBAL_CHAINS",
    "seed": "EB_GLOBAL_SEED",
    "ksi_mode": "EB_GLOBAL_KSI_MODE",
    "prior_dirichlet": "EB_GLOBAL_PRIOR_DIRICHLET",
    "prior_dirichlet_rp": "EB_GLOBAL_PRIOR_DIRICHLET_RP",
    "prior_ksi_sigma": "EB_GLOBAL_PRIOR_KSI_SIGMA",
    "fixed_ksi": "EB_GLOBAL_FIXED_KSI",
    "prior_ksi_ig_alpha": "EB_GLOBAL_PRIOR_KSI_IG_ALPHA",
    "prior_ksi_ig_beta": "EB_GLOBAL_PRIOR_KSI_IG_BETA",
    "prior_ig_alpha": "EB_GLOBAL_PRIOR_IG_ALPHA",
    "prior_ig_beta": "EB_GLOBAL_PRIOR_IG_BETA",
    "delta_bounds": "EB_GLOBAL_DELTA_BOUNDS",
    "tk_delta_bounds": "EB_GLOBAL_TK_DELTA_BOUNDS",
    "prelec_delta_bounds": "EB_GLOBAL_PRELEC_DELTA_BOUNDS",
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


@dataclass
class RunningJob:
    job: dict
    process: subprocess.Popen
    log_file: object
    started_at: float


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_settings(path):
    settings_path = Path(path)
    if not settings_path.is_absolute():
        settings_path = ROOT / settings_path
    spec = importlib.util.spec_from_file_location("batch_settings", settings_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load settings file: {settings_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, settings_path


def _slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value)).strip("-._")
    return slug or "run"


def _lottery_label(job):
    lottery_ids = job.get("lottery_ids")
    if lottery_ids:
        if isinstance(lottery_ids, str):
            ids = [item.strip() for item in lottery_ids.split(",") if item.strip()]
        else:
            ids = [str(item) for item in lottery_ids]
        if len(ids) <= 4:
            return "ids-" + "-".join(ids)
        return f"ids-{len(ids)}"
    return str(job.get("lottery_set", "full"))


def _bounds_label(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]

    if len(parts) == 2:
        try:
            lower, upper = float(parts[0]), float(parts[1])
            if lower == upper:
                return _slugify(f"fixed-{lower:g}")
            return _slugify(f"{lower:g}-{upper:g}")
        except (TypeError, ValueError):
            pass

    return _slugify("-".join(str(part) for part in parts))


def _delta_bounds_label(job):
    method = str(job.get("method", "")).lower().replace("-", "_")
    for key in (f"{method}_delta_bounds", "delta_bounds"):
        if key in job:
            return _bounds_label(job.get(key))
    return None


def _default_name(job):
    method = job.get("method", "method")
    discounting = job.get("discounting", "discount")
    clusters = int(job.get("clusters", 0))
    ksi_mode = job.get("ksi_mode", "ksi")
    lottery_label = _lottery_label(job)
    delta_label = _delta_bounds_label(job)
    delta_part = f"_delta-{delta_label}" if delta_label else ""
    draws = job.get("draws", "draws")
    chains = job.get("chains", "chains")
    seed = job.get("seed", "seed")
    return (
        f"{method}_{discounting}_C{clusters:02d}_lotteries-{lottery_label}_"
        f"ksi-{ksi_mode}{delta_part}_d{draws}_ch{chains}_s{seed}"
    )


def _env_value(value):
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _requested_cores(job):
    chains = int(job.get("chains", 1))
    raw = job.get("cores", job.get("smc_cores", chains))
    if raw in (None, "") or str(raw).lower() in {"none", "null", "auto"}:
        raw = chains
    return max(1, min(chains, int(raw)))


def _set_pytensor_flag(flags, key, value):
    parts = [
        part.strip()
        for part in flags.split(",")
        if part.strip() and not part.strip().startswith(f"{key}=")
    ]
    parts.append(f"{key}={value}")
    return ",".join(parts)


def _normalise_jobs(settings, settings_path, max_cores, only):
    output_root = Path(getattr(settings, "OUTPUT_ROOT", "batch_outputs"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    defaults = dict(getattr(settings, "DEFAULTS", {}))
    extra_env = dict(getattr(settings, "EXTRA_ENV", {}))
    runs = list(getattr(settings, "RUNS", []))
    jobs = []
    seen = {}

    for index, run in enumerate(runs, start=1):
        job = dict(defaults)
        job.update(run)
        if not job.get("enabled", True):
            continue

        name = _slugify(job.get("name") or _default_name(job))
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}-{seen[name]}"

        if only and not any(token in name for token in only):
            continue

        output_dir = Path(job.get("output_dir") or output_root / name)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir

        allocated_cores = min(_requested_cores(job), max_cores)
        job.update(
            {
                "_index": index,
                "_name": name,
                "_settings_file": str(settings_path),
                "_output_dir": str(output_dir),
                "_allocated_cores": allocated_cores,
                "_extra_env": extra_env,
            }
        )
        jobs.append(job)

    return jobs


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _is_completed(job):
    output_dir = Path(job["_output_dir"])
    status_path = output_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            return status.get("status") == "completed"
        except json.JSONDecodeError:
            pass
    return (
        (output_dir / "trace.nc").exists()
        and (output_dir / "summary.csv").exists()
        and (output_dir / "summary_reference_weights.csv").exists()
    )


def _job_env(job):
    output_dir = Path(job["_output_dir"])
    env = os.environ.copy()

    for key, env_key in ENV_KEYS.items():
        value = job.get(key)
        if value is not None:
            env[env_key] = _env_value(value)

    env["EB_GLOBAL_OUTPUT_DIR"] = str(output_dir)
    env["EB_GLOBAL_SMC_CORES"] = str(job["_allocated_cores"])
    env["EB_GLOBAL_PROGRESSBAR"] = "1" if _as_bool(job.get("progressbar", False)) else "0"

    thread_limits = job.get("thread_limits") or {}
    for key, value in THREAD_LIMITS.items():
        env[key] = str(thread_limits.get(key, value))

    for key, value in job.get("_extra_env", {}).items():
        env[key] = str(value)
    for key, value in job.get("env", {}).items():
        env[key] = str(value)

    env["MPLCONFIGDIR"] = str(output_dir / "_matplotlib")
    env["PYTENSOR_FLAGS"] = _set_pytensor_flag(
        env.get("PYTENSOR_FLAGS", ""),
        "base_compiledir",
        str(output_dir / "_pytensor"),
    )
    return env


def _append_batch_status(status_csv, row):
    fieldnames = [
        "finished_at",
        "status",
        "returncode",
        "name",
        "allocated_cores",
        "elapsed_seconds",
        "output_dir",
        "log_path",
    ]
    exists = status_csv.exists()
    with status_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _launch(job, python_executable):
    output_dir = Path(job["_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_matplotlib").mkdir(exist_ok=True)
    (output_dir / "_pytensor").mkdir(exist_ok=True)

    settings_payload = {
        key: value
        for key, value in job.items()
        if not key.startswith("_") or key in {"_name", "_allocated_cores", "_settings_file"}
    }
    _write_json(output_dir / "settings.json", settings_payload)
    _write_json(
        output_dir / "status.json",
        {
            "name": job["_name"],
            "status": "running",
            "started_at": _now(),
            "allocated_cores": job["_allocated_cores"],
        },
    )

    command = [python_executable, "main.py"]
    if _as_bool(job.get("load", False)):
        command.append("--load")

    log_path = output_dir / "run.log"
    log_file = log_path.open("a", buffering=1)
    log_file.write(f"\n=== START {_now()} ===\n")
    log_file.write(f"command: {' '.join(command)}\n")
    log_file.write(f"output_dir: {output_dir}\n\n")

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=_job_env(job),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(
        f"[start] {job['_name']} "
        f"pid={process.pid} cores={job['_allocated_cores']} -> {output_dir}"
    )
    return RunningJob(job=job, process=process, log_file=log_file, started_at=time.time())


def _finish(running, status_csv):
    rc = running.process.returncode
    elapsed = round(time.time() - running.started_at, 1)
    status = "completed" if rc == 0 else "failed"
    output_dir = Path(running.job["_output_dir"])
    log_path = output_dir / "run.log"

    running.log_file.write(f"\n=== END {_now()} returncode={rc} ===\n")
    running.log_file.close()

    payload = {
        "name": running.job["_name"],
        "status": status,
        "returncode": rc,
        "started_at": datetime.fromtimestamp(running.started_at).astimezone().isoformat(timespec="seconds"),
        "finished_at": _now(),
        "elapsed_seconds": elapsed,
        "allocated_cores": running.job["_allocated_cores"],
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }
    _write_json(output_dir / "status.json", payload)
    _append_batch_status(status_csv, payload)
    print(f"[{status}] {running.job['_name']} rc={rc} elapsed={elapsed}s")


def _terminate_all(running_jobs, status_csv):
    for running in running_jobs:
        if running.process.poll() is None:
            running.process.terminate()
    time.sleep(2)
    for running in running_jobs:
        if running.process.poll() is None:
            running.process.kill()
        running.process.wait()
        _finish(running, status_csv)


def _run_scheduler(jobs, max_cores, python_executable, poll_seconds, skip_completed):
    if not jobs:
        print("No enabled jobs to run.")
        return

    status_csv = Path(jobs[0]["_output_dir"]).parent / "_batch_status.csv"
    pending = list(jobs)
    running = []
    used_cores = 0

    try:
        while pending or running:
            launched = True
            while launched:
                launched = False
                for job in list(pending):
                    if skip_completed and _is_completed(job):
                        print(f"[skip] {job['_name']} already completed")
                        pending.remove(job)
                        launched = True
                        break
                    if used_cores + job["_allocated_cores"] <= max_cores:
                        running_job = _launch(job, python_executable)
                        running.append(running_job)
                        pending.remove(job)
                        used_cores += job["_allocated_cores"]
                        launched = True
                        break

            time.sleep(poll_seconds)
            for running_job in list(running):
                if running_job.process.poll() is None:
                    continue
                _finish(running_job, status_csv)
                used_cores -= running_job.job["_allocated_cores"]
                running.remove(running_job)

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active jobs...")
        _terminate_all(running, status_csv)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian CPT batch jobs.")
    parser.add_argument("--settings", default="BatchSettings.py", help="Python settings file")
    parser.add_argument("--max-cores", type=int, default=None, help="override settings MAX_CORES")
    parser.add_argument("--dry-run", action="store_true", help="show queue without launching jobs")
    parser.add_argument("--rerun", action="store_true", help="run even if status/output files already exist")
    parser.add_argument("--only", action="append", default=[], help="run only jobs whose name contains this text")
    args = parser.parse_args()

    settings, settings_path = _load_settings(args.settings)
    max_cores = args.max_cores or int(getattr(settings, "MAX_CORES", os.cpu_count() or 1))
    poll_seconds = float(getattr(settings, "POLL_SECONDS", 5))
    python_executable = getattr(settings, "PYTHON_EXECUTABLE", None) or sys.executable
    skip_completed = bool(getattr(settings, "SKIP_COMPLETED", True)) and not args.rerun

    jobs = _normalise_jobs(settings, settings_path, max_cores, args.only)

    print(f"settings: {settings_path}")
    print(f"max cores: {max_cores}")
    print(f"jobs: {len(jobs)}")
    for job in jobs:
        print(
            f"  {job['_name']:<40} "
            f"cores={job['_allocated_cores']:<2} "
            f"out={job['_output_dir']}"
        )

    if args.dry_run:
        return

    _run_scheduler(
        jobs=jobs,
        max_cores=max_cores,
        python_executable=python_executable,
        poll_seconds=poll_seconds,
        skip_completed=skip_completed,
    )


if __name__ == "__main__":
    main()
