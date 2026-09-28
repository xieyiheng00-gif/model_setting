"""GPU telemetry: the NVML sampler (with a fake NVML), its per-step summary, the peak-FLOPs table used for the live
MFU, and scripts/gpu_report.py on a small synthetic run. No GPU needed."""
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

from lmarch.train import gpu_monitor as G


def _fake_nvml():
    nv = types.ModuleType("pynvml")
    nv.NVML_TEMPERATURE_GPU, nv.NVML_CLOCK_SM = 0, 1
    nv.nvmlInit = nv.nvmlShutdown = lambda: None
    nv.nvmlDeviceGetCount = lambda: 2
    nv.nvmlDeviceGetHandleByIndex = lambda i: i
    nv.nvmlDeviceGetName = lambda h: b"NVIDIA H100 80GB HBM3"
    nv.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=90 + 2 * h, memory=40)
    nv.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=(20 + h) * 2**30, total=80 * 2**30)

    def power(h):
        if h == 1:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return 600_000
    nv.nvmlDeviceGetPowerUsage = power
    nv.nvmlDeviceGetTemperature = lambda h, sensor: 60 + h
    nv.nvmlDeviceGetClockInfo = lambda h, clock: 1980 - 225 * h
    nv.nvmlDeviceGetCurrentClocksEventReasons = lambda h: 0x4 if h == 1 else 0x0
    return nv


def test_monitor_samples_every_gpu_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    mon = G.GPUMonitor(tmp_path / "metrics" / "gpu.jsonl", 0.01, lambda: 7)
    deadline = time.time() + 5
    while time.time() < deadline and len(mon._window) < 3:
        time.sleep(0.01)
    s = mon.summary()
    mon.close()
    assert mon.names == ["NVIDIA H100 80GB HBM3"] * 2
    assert s["samples"] >= 3 and s["util_mean"] == pytest.approx(91) and s["util_min_gpu"] == 90
    assert s["power_w_mean"] == 600 and s["sm_mhz_min"] == 1755 and s["mem_gb_max"] == 21 and s["temp_c_max"] == 61
    assert s["power_capped_frac"] == pytest.approx(0.5) and s["thermal_throttled_frac"] == 0
    recs = [json.loads(line) for line in (tmp_path / "metrics" / "gpu.jsonl").read_text().splitlines()]
    assert recs and recs[0]["step"] == 7 and [g["i"] for g in recs[0]["gpus"]] == [0, 1]
    assert recs[0]["gpus"][1]["power_w"] is None                 # unsupported query: recorded as missing
    assert mon.summary() is None or mon.summary()["samples"] >= 0


def test_monitor_reads_performance_counters(tmp_path, monkeypatch):
    """SM active / occupancy, tensor and DRAM activity through the real nvidia-ml-py ctypes structures (no GPU:
    the NVML calls are faked); GPU 1 has no counters and must simply lack them."""
    real = pytest.importorskip("pynvml")
    nv = _fake_nvml()
    for name in ("c_nvmlGpmMetricsGet_t", "c_nvmlGpmSample_t", "NVML_GPM_METRICS_GET_VERSION", "NVML_GPM_METRIC_SM_UTIL",
                 "NVML_GPM_METRIC_SM_OCCUPANCY", "NVML_GPM_METRIC_ANY_TENSOR_UTIL", "NVML_GPM_METRIC_DRAM_BW_UTIL"):
        setattr(nv, name, getattr(real, name))
    values = {real.NVML_GPM_METRIC_SM_UTIL: 88.0, real.NVML_GPM_METRIC_SM_OCCUPANCY: 30.0,
              real.NVML_GPM_METRIC_ANY_TENSOR_UTIL: 41.5, real.NVML_GPM_METRIC_DRAM_BW_UTIL: 55.0}
    nv.nvmlGpmQueryDeviceSupport = lambda h: types.SimpleNamespace(isSupportedDevice=int(h == 0))
    nv.nvmlGpmSampleAlloc = lambda: real.c_nvmlGpmSample_t()
    nv.nvmlGpmSampleGet = lambda h, smp: smp
    nv.nvmlGpmSampleFree = lambda smp: None

    def metrics_get(mg):
        assert mg.version == real.NVML_GPM_METRICS_GET_VERSION and mg.numMetrics == 4
        for j in range(mg.numMetrics):
            mg.metrics[j].value = values[mg.metrics[j].metricId]
            mg.metrics[j].nvmlReturn = 0
        return mg
    nv.nvmlGpmMetricsGet = metrics_get
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    mon = G.GPUMonitor(tmp_path / "gpu.jsonl", 0.01, lambda: 3)
    deadline = time.time() + 5
    while time.time() < deadline and len(mon._window) < 3:
        time.sleep(0.01)
    s = mon.summary()
    mon.close()
    assert mon.gpm_status == "on (1 of 2 GPUs)"
    assert s["sm_active_mean"] == 88.0 and s["sm_occupancy_mean"] == 30.0
    assert s["tensor_active_mean"] == 41.5 and s["dram_active_mean"] == 55.0
    rec = json.loads((tmp_path / "gpu.jsonl").read_text().splitlines()[-1])
    assert rec["gpus"][0]["tensor_active"] == 41.5 and "tensor_active" not in rec["gpus"][1]


def test_summary_of_nothing_is_none():
    assert G.summarize([]) is None


def test_peak_flops_by_gpu_name():
    assert G.peak_flops("NVIDIA H100 80GB HBM3") == pytest.approx(989.4e12)
    assert G.peak_flops("NVIDIA H100 PCIe") == pytest.approx(756e12)
    assert G.peak_flops("NVIDIA GeForce RTX 3060") == 0.0            # unknown: no MFU unless log.peak_tflops
    assert G.peak_flops("NVIDIA GeForce RTX 3060", override_tflops=25.6) == pytest.approx(25.6e12)


def test_wandb_gets_gpu_and_mfu_panels(tmp_path, monkeypatch):
    logged, init_kw = [], {}
    run = types.SimpleNamespace(define_metric=lambda *a, **k: None, log=logged.append, url="u",
                                finish=lambda **k: None)
    fake = types.ModuleType("wandb")
    fake.util = types.SimpleNamespace(generate_id=lambda: "abc")
    fake.Settings = lambda **kw: kw

    def init(**kw):
        init_kw.update(kw)
        return run
    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    from lmarch.config import LogConfig
    from lmarch.train.wandb_sink import WandbSink
    sink = WandbSink(LogConfig(wandb=True, wandb_mode="offline"), tmp_path, "r", {}, [], "g", "trunk")
    assert init_kw["settings"]["x_stats_sampling_interval"] == 5.0      # W&B system tab: every 5 s, not 15
    sink.train({"step": 3, "loss": 2.0, "mfu": 0.21, "gpu": {"util_mean": 97.0, "power_w_mean": 610.0},
                "reasons": []})
    p = logged[-1]
    assert p["train/mfu"] == 0.21 and p["gpu/util_mean"] == 97.0 and p["gpu/power_w_mean"] == 610.0
    assert p["trainer/step"] == 3


def _load_report():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_report.py"
    spec = importlib.util.spec_from_file_location("gpu_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gpu_report_on_a_synthetic_run(tmp_path):
    rd = tmp_path / "kda_full_trunk_h100x8"
    m = rd / "metrics"
    m.mkdir(parents=True)
    t0 = 1_790_000_000.0
    with open(m / "gpu.jsonl", "w") as f:
        for s in range(600):                                        # 10 min, 2 GPUs; a 30 s stall at 5 min
            stall = 300 <= s < 330
            gpus = [{"i": i, "util": 0 if stall else 98 - 4 * i, "power_w": 150 if stall else 600, "sm_mhz": 1980,
                     "mem_gb": 20, "temp_c": 60, "throttle": 0x4 if (s % 10 == 0 and not stall) else 0,
                     "sm_active": 0 if stall else 80, "tensor_active": 0 if stall else 40}
                    for i in range(2)]
            f.write(json.dumps({"time": t0 + s, "step": s, "gpus": gpus}) + "\n")
    with open(m / "train_steps.jsonl", "w") as f:
        for s in range(5, 590):
            f.write(json.dumps({"step": s, "time": t0 + s, "step_time": 1.0, "tokens_per_sec": 5e5, "mfu": 0.2}) + "\n")
    (m / "events.jsonl").write_text(json.dumps({"time": t0 + 330, "kind": "checkpoint", "seconds": 30.0}) + "\n")
    rep = _load_report()
    out = tmp_path / "report"
    assert rep.main([str(rd), "--out", str(out), "--no-plots"]) == 0
    row = json.loads((out / "gpu_summary.json").read_text())[0]
    assert row["gpus"] == 2 and row["slowest_gpu"] == 1 and row["low_util_min"] == pytest.approx(0.5)
    assert row["mfu_median_pct"] == 20.0 and row["power_capped_pct"] == pytest.approx(9.5)   # 57 of 600 s
    assert row["sm_active_mean"] == pytest.approx(76.0) and row["tensor_active_mean"] == pytest.approx(38.0)
    assert "dram_active_mean" not in row                             # a counter that was never recorded
    stretch = row["low_util_stretches"][0]
    assert stretch["minutes"] == pytest.approx(0.5) and "checkpoint save" in stretch["during"]
    if importlib.util.find_spec("matplotlib"):
        assert rep.main([str(rd), "--out", str(out)]) == 0
        assert (out / "kda_full_trunk_h100x8_gpu.png").stat().st_size > 10_000
