"""Milestone-based Phase A orchestrator (reviewer round-2 revision).

Runs the post-embedding sequence from REVIEWER_CHECK.md, in order, with
retries and a resumable state file. Also supervises the embedding jobs
themselves: if a tag's embed_all process is not running and its final
concat is missing, it relaunches that tag (staggered), so a crash or a
reboot mid-embedding self-heals when the runner is started again.

State: cache/phase_a_state.json  ->  {"done": ["embedding", ...], "attempts": {...}}
Log:   logs/phase_a.log

Usage: python -u scripts/phase_a_runner.py [--dry-run] [--retries N]
Safe to run repeatedly / after a reboot; completed milestones are skipped.
"""
import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import EWS_MODEL_DIR, ABLATE_DIR, EMBED_DIR, OUTPUT_DIR  # noqa: E402

STATE = PROJECT / "cache" / "phase_a_state.json"
LOG = PROJECT / "logs" / "phase_a.log"
NOTEBOOK_TMP = Path("/tmp/pipeline_check.ipynb")

FINAL_FILES = [
    EMBED_DIR / "dev_window_ecgfm.npy", EMBED_DIR / "dev_window_papagei.npy",
    EMBED_DIR / "dev_meta.csv", EMBED_DIR / "test_window_ecgfm.npy",
    EMBED_DIR / "test_window_papagei.npy", EMBED_DIR / "test_meta.csv",
]

# (name, command, timeout_s, done_files or None = always run unless in state)
MILESTONES = [
    ("train_fean",
     ["python", "scripts/train_fean_models.py"], 5400,
     [EWS_MODEL_DIR / "model.pt", EWS_MODEL_DIR / "stats.json"]),
    ("train_ablations",
     ["python", "scripts/train_ablations.py"], 5400,
     [ABLATE_DIR / "val_aucs.json"]),
    ("notebook",
     ["jupyter", "nbconvert", "--to", "notebook", "--execute",
      "Pipeline.ipynb", "--output", str(NOTEBOOK_TMP),
      "--ExecutePreprocessor.timeout=7200"], 7800, None),
    ("test_fusion",
     ["python", "scripts/test_fusion.py"], 3600, None),
    ("test_fean_v2",
     ["python", "scripts/test_fean_v2.py"], 3600, None),
    ("robustness",
     ["python", "scripts/robustness.py"], 5400,
     [OUTPUT_DIR / "robustness_metrics.json"]),
    ("sddb_zero_shot",
     ["python", "scripts/analyze_sddb.py", "--zero-shot-only",
      "--model-dir", "cache/models/ews_v2"], 3600, None),
]

# files must be NEWER than the 240-patient cohort build, or they are
# stale artifacts from the old 60-patient run (e.g. robustness_metrics.json)
COHORT_MTIME = (PROJECT / "cache" / "splits.json").stat().st_mtime


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": [], "attempts": {}}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def final_files_present():
    return all(p.exists() for p in FINAL_FILES)


def embed_running(tag):
    try:
        res = subprocess.run(["pgrep", "-f", f"embed_all.py --tag {tag}"],
                             capture_output=True, text=True, timeout=10)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return True    # assume running rather than spawn a duplicate


def launch_embed(tag):
    """Detached relaunch of one embedding tag (log appended)."""
    log(f"relaunching embedding tag {tag}")
    out = open(PROJECT / "logs" / f"embed_{tag}.log", "ab")
    subprocess.Popen([sys.executable, "-u", "scripts/embed_all.py",
                      "--tag", tag], stdout=out, stderr=subprocess.STDOUT,
                     cwd=PROJECT, start_new_session=True)


def supervise_embedding():
    """Wait for the final concats, relaunching dead tags as needed."""
    while not final_files_present():
        try:
            missing = [t for t in ("dev", "test") if not embed_running(t)]
            if missing:
                for t in missing:
                    launch_embed(t)
                    time.sleep(180)      # stagger segment-build memory spikes
        except Exception as e:           # never let the supervise loop die
            log(f"supervise loop error: {e!r} (continuing)")
        log(f"embedding not done yet; "
            f"groups: {sum(1 for p in EMBED_DIR.glob('*_g*_window_ecgfm.npy'))}")
        time.sleep(300)
    log("embedding complete: all final files present")


def milestone_done(state, name, done_files):
    if name in state["done"]:
        return True
    if done_files and all(p.exists() and p.stat().st_mtime > COHORT_MTIME
                          for p in done_files):
        return True
    return False


def run_milestone(state, name, cmd, timeout_s, done_files, retries):
    if milestone_done(state, name, done_files):
        log(f"[skip] {name} already done")
        return True
    for attempt in range(1, retries + 1):
        log(f"[start] {name} (attempt {attempt}/{retries})")
        t0 = time.time()
        try:
            res = subprocess.run(cmd, cwd=PROJECT, timeout=timeout_s,
                                 capture_output=True, text=True)
            tail = (res.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                log(f"  {name} | {line}")
            if res.stderr:
                log(f"  {name} | stderr: {res.stderr.strip().splitlines()[-3:]}")
            ok = res.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
            log(f"  {name} | TIMEOUT after {timeout_s}s")
        if ok:
            if name == "notebook" and NOTEBOOK_TMP.exists():
                shutil.copy(NOTEBOOK_TMP, PROJECT / "Pipeline.ipynb")
                log("  notebook | executed notebook copied to Pipeline.ipynb")
            state["done"].append(name)
            save_state(state)
            log(f"[done] {name} in {time.time() - t0:.0f}s")
            return True
        log(f"[fail] {name} attempt {attempt} failed; "
            f"retrying in 120s" if attempt < retries else
            f"[ABORT] {name} failed {retries} times")
        time.sleep(120)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print milestone status without executing")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    (PROJECT / "logs").mkdir(exist_ok=True)

    # single-instance lock: two runners must never execute milestones at once
    lock_fd = open(PROJECT / "logs" / "phase_a.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another phase_a_runner is already running; exiting")
        sys.exit(0)

    state = load_state()
    log(f"phase_a_runner start (dry-run={args.dry_run}, "
        f"done={state['done']})")

    if args.dry_run:
        for name, cmd, tmo, done in MILESTONES:
            status = "DONE" if milestone_done(state, name, done) else "pending"
            print(f"  {name:16s} {status}")
        print(f"  embedding        "
              f"{'DONE' if final_files_present() else 'pending'}")
        return

    if "embedding" not in state["done"]:
        if not final_files_present():
            supervise_embedding()
        state["done"].append("embedding")
        save_state(state)

    for name, cmd, tmo, done in MILESTONES:
        if not run_milestone(state, name, cmd, tmo, done, args.retries):
            log(f"ABORTED at {name}; fix the failure and re-run the runner "
                f"(completed milestones are cached).")
            sys.exit(1)

    log("PHASE A COMPLETE: all milestones done.")
    log("next manual step: refresh manuscript tables "
        "(TODO(PHASE-A) markers in ../MDPI_Hybrid_Revision/main.tex)")


if __name__ == "__main__":
    main()
