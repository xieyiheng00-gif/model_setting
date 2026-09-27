"""GPU telemetry for the whole run, plus the peak-FLOPs table used for live MFU.

On rank 0 a background thread samples every GPU of the machine through NVML (nvidia-ml-py) every
log.gpu_interval_sec (default 1 s) and appends one line per sample to metrics/gpu.jsonl:
    {"time": ..., "step": ..., "gpus": [{"i": 0, "util": %, "mem_util": %, "mem_gb": ..., "power_w": ...,
                                         "temp_c": ..., "sm_mhz": ..., "throttle": NVML clock-event bitmask}, ...]}
Every training step record also gets a summary of the samples taken since the previous step ("gpu": mean and
lowest-GPU utilization, power, lowest SM clock, peak memory and temperature, share of throttled samples). W&B shows
it as gpu/* on the same trainer/step axis as the loss, and the JSONL goes to the Hub with every checkpoint upload.
`python scripts/gpu_report.py runs/<run>` turns it into a summary and plots after training.

"Utilization" is NVML's share of time in which at least one kernel was running. It says nothing about how much
of the GPU those kernels use (the 1xH100 speed test: 99.7% utilization at 12.7% MFU). For efficiency look at the
trainer's `mfu` and `tokens_per_sec`, the power draw, and W&B's system metrics smActive / pipeTensorActive.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# NVML clock-event ("throttle") reasons that mean the GPU runs below its maximum clock for a hardware reason
THROTTLE_POWER = 0x4 | 0x80        # SW power cap, HW power brake
THROTTLE_THERMAL = 0x20 | 0x40     # SW / HW thermal slowdown
THROTTLE_SLOWDOWN = 0x8            # HW slowdown

# dense BF16 tensor-core peak (no sparsity), by device-name substring; first match wins
PEAK_BF16_TFLOPS = (
    ("H100 PCIe", 756.0), ("H100 NVL", 835.0), ("H100", 989.4), ("H200", 989.4),
    ("A100", 312.0), ("A800", 312.0), ("L40S", 362.0), ("RTX 4090", 165.2), ("RTX 3090", 71.0),
)


def peak_flops(device_name: str, override_tflops: float = 0.0) -> float:
    """Peak BF16 FLOP/s of one GPU: log.peak_tflops if set, else the table above; 0 = unknown (no MFU)."""
    if override_tflops > 0:
        return override_tflops * 1e12
    for key, tf in PEAK_BF16_TFLOPS:
        if key in device_name:
            return tf * 1e12
    return 0.0


class GPUMonitor:
    def __init__(self, path: Path, interval_s: float, step_fn):
        import pynvml                                # nvidia-ml-py; ImportError -> the caller turns the monitor off
        pynvml.nvmlInit()
        self.nv = pynvml
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.names = []
        for h in self.handles:
            n = pynvml.nvmlDeviceGetName(h)
            self.names.append(n.decode() if isinstance(n, bytes) else str(n))
        self._reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        self.path, self.interval, self.step_fn = Path(path), float(interval_s), step_fn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._window: list[dict] = []
        self._stop = threading.Event()
        self._f = open(self.path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def _read(self, i: int, h) -> dict:
        nv = self.nv
        g: dict = {"i": i}

        def q(fn):
            try:
                return fn()
            except Exception:                        # a query this GPU/driver does not support
                return None
        u = q(lambda: nv.nvmlDeviceGetUtilizationRates(h))
        g["util"] = None if u is None else u.gpu
        g["mem_util"] = None if u is None else u.memory
        m = q(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        g["mem_gb"] = None if m is None else round(m.used / 2**30, 2)
        p = q(lambda: nv.nvmlDeviceGetPowerUsage(h))
        g["power_w"] = None if p is None else round(p / 1000.0, 1)
        g["temp_c"] = q(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        g["sm_mhz"] = q(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
        g["throttle"] = q(lambda: int(self._reasons(h))) if self._reasons else None
        return g

    def sample(self) -> dict:
        return {"time": round(time.time(), 3), "step": self.step_fn(),
                "gpus": [self._read(i, h) for i, h in enumerate(self.handles)]}

    def _loop(self) -> None:
        while True:
            try:
                rec = self.sample()
                self._f.write(json.dumps(rec) + "\n")
                self._f.flush()
                with self._lock:
                    self._window.append(rec)
            except Exception:                        # never let telemetry take the run down
                pass
            if self._stop.wait(self.interval):
                return

    def summary(self) -> dict | None:
        """Aggregate of the samples since the previous call (None if there were none)."""
        with self._lock:
            win, self._window = self._window, []
        return summarize(win)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._f.close()
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


def summarize(samples: list[dict]) -> dict | None:
    """Per-step summary of telemetry samples: mean utilization over GPUs and samples, the lowest per-GPU mean
    (one straggling GPU), power, lowest SM clock, peak memory/temperature and the share of throttled GPU-samples."""
    if not samples:
        return None
    per_gpu: dict[int, list[float]] = {}
    util, power, sm, mem, temp, thr_p, thr_t, n_thr = [], [], [], [], [], 0, 0, 0
    for s in samples:
        for g in s["gpus"]:
            if g.get("util") is not None:
                util.append(g["util"])
                per_gpu.setdefault(g["i"], []).append(g["util"])
            for lst, k in ((power, "power_w"), (sm, "sm_mhz"), (mem, "mem_gb"), (temp, "temp_c")):
                if g.get(k) is not None:
                    lst.append(g[k])
            if g.get("throttle") is not None:
                n_thr += 1
                thr_p += bool(g["throttle"] & THROTTLE_POWER)
                thr_t += bool(g["throttle"] & (THROTTLE_THERMAL | THROTTLE_SLOWDOWN))
    out: dict = {"samples": len(samples)}
    if util:
        out["util_mean"] = sum(util) / len(util)
        out["util_min_gpu"] = min(sum(v) / len(v) for v in per_gpu.values())
    if power:
        out["power_w_mean"] = sum(power) / len(power)
    if sm:
        out["sm_mhz_min"] = min(sm)
    if mem:
        out["mem_gb_max"] = max(mem)
    if temp:
        out["temp_c_max"] = max(temp)
    if n_thr:
        out["power_capped_frac"] = thr_p / n_thr
        out["thermal_throttled_frac"] = thr_t / n_thr
    return out
