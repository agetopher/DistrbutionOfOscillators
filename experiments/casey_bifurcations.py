"""Regenerate the focused Casey handoff at the canonical 400 ms baseline."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bifurcations as bif
import drive_snic
import settings
import segment

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    started = datetime.now(timezone.utc).isoformat()
    settings.reset_defaults()
    settings.tau_w = 400.0
    settings.beta = 1.03
    segment.configure()
    MEDIA.mkdir(exist_ok=True)
    parameters = {
        "tau_w_ms": settings.tau_w, "beta_nS_per_mV": settings.beta,
        "k_syn_per_mV": settings.k_syn,
        "G_syne_nS": settings.G_syne, "G_syni_nS": settings.G_syni,
        "G_gap_nS": settings.G_gap,
    }
    timings = {}
    print(f"Baseline: {parameters}", flush=True)
    for name in ("snic", "drive_stability", "beta_stability"):
        before = perf_counter()
        print(f"Starting {name}", flush=True)
        if name == "snic":
            drive_snic.run(beta=1.03, tau_w=400.0)
        elif name == "drive_stability":
            records = [bif.analyze_stability(branch) for branch in ("AVA", "AVB")]
            bif.plot_stability(records, save=True)
        else:
            records = [bif.analyze_beta(branch, drive=2.5)
                       for branch in ("AVA", "AVB")]
            bif.plot_beta_stability(records, save=True)
        if name != "snic":
            (MEDIA / f"bifurcation_{name}_results.json").write_text(
                json.dumps(serializable({"parameters": parameters,
                    "beta_is_swept": name == "beta_stability",
                    "records": records}), indent=2, allow_nan=False) + "\n")
        plt.close("all")
        timings[name] = perf_counter() - before
        print(f"Finished {name}: {timings[name]:.1f} seconds", flush=True)
    outputs = [
        "bifurcation_drive_snic_equilibrium.png", "bifurcation_drive_snic_cycles.png",
        "bifurcation_drive_snic_results.json", "bifurcation_stability_eigen.png",
        "bifurcation_drive_stability_results.json", "bifurcation_beta_stability.png",
        "bifurcation_beta_stability_results.json",
    ]
    manifest = {"started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": parameters, "elapsed_seconds": timings,
                "beta_sweep": {"min": 0.8, "max": 2.0, "step": 0.02,
                               "active_drive_pA": 2.5, "other_drive_pA": 0},
                "sha256": {name: hashlib.sha256((MEDIA / name).read_bytes()).hexdigest()
                           for name in outputs}}
    (MEDIA / "casey_bifurcations_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
