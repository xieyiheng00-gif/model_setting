"""Raw H100 microbenchmarks vs the NVIDIA H100 SXM5 datasheet.

GEMM throughput (BF16/FP16/TF32/FP8), HBM bandwidth, host<->device PCIe bandwidth, and clocks/power
sampled with nvidia-smi during a sustained BF16 GEMM. Writes gpu_bench.json next to this file.
"""
import json, os, subprocess, threading, time
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_bench.json")
dev = torch.device("cuda")

# NVIDIA H100 SXM5 80GB datasheet peaks (dense, no sparsity)
SPEC = {"bf16_tflops": 989.4, "fp16_tflops": 989.4, "tf32_tflops": 494.7, "fp8_tflops": 1978.9,
        "hbm_tbps": 3.35, "pcie_gen5_x16_gbps_per_dir": 64.0, "tdp_w": 700, "boost_mhz": 1980}


def timed(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters


def gemm_tflops(n, dtype, iters=50):
    if dtype == "fp8":
        a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()  # column-major for _scaled_mm
        one = torch.tensor(1.0, device=dev)
        fn = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    else:
        torch.backends.cuda.matmul.allow_tf32 = dtype == torch.float32
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        fn = lambda: a @ b
    t = timed(fn, iters)
    return 2 * n ** 3 / t / 1e12


def hbm_bw():
    n = 2 ** 30  # 1 Gi floats = 4 GiB per buffer
    x = torch.empty(n, device=dev, dtype=torch.float32).uniform_()
    y = torch.empty_like(x)
    t_copy = timed(lambda: y.copy_(x), 30)
    t_read = timed(lambda: x.sum(), 30)
    return {"copy_tbps": 2 * x.numel() * 4 / t_copy / 1e12, "read_tbps": x.numel() * 4 / t_read / 1e12}


def pcie_bw():
    n = 2 ** 28  # 1 GiB
    h = torch.empty(n, dtype=torch.float32).pin_memory()
    d = torch.empty(n, dtype=torch.float32, device=dev)
    h2d = timed(lambda: d.copy_(h, non_blocking=True), 10, 2)
    d2h = timed(lambda: h.copy_(d, non_blocking=True), 10, 2)
    return {"h2d_gbps": n * 4 / h2d / 1e9, "d2h_gbps": n * 4 / d2h / 1e9}


def sustained_bf16(seconds=60, n=8192):
    """Run back-to-back BF16 GEMMs for `seconds`, sampling clocks/power/temp every 0.5 s."""
    samples, stop = [], threading.Event()

    def poll():
        q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
        while not stop.is_set():
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(", ")
            try:
                samples.append({"sm_mhz": float(r[0]), "mem_mhz": float(r[1]), "power_w": float(r[2]),
                                "temp_c": float(r[3]), "throttle": r[4]})
            except (ValueError, IndexError):
                pass
            time.sleep(0.5)

    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    per_window, t0, count = [], time.time(), 0
    while time.time() - t0 < seconds:
        tw = time.time()
        for _ in range(100):
            a @ b
        torch.cuda.synchronize()
        dt = time.time() - tw
        per_window.append(100 * 2 * n ** 3 / dt / 1e12)
        count += 100
    total = time.time() - t0
    stop.set()
    th.join()
    steady = samples[len(samples) // 5:] or samples  # drop the ramp-up
    avg = lambda k: sum(s[k] for s in steady) / len(steady)
    return {"seconds": total, "gemms": count, "avg_tflops": count * 2 * n ** 3 / total / 1e12,
            "min_window_tflops": min(per_window), "max_window_tflops": max(per_window),
            "avg_sm_mhz": avg("sm_mhz"), "min_sm_mhz": min(s["sm_mhz"] for s in steady),
            "avg_power_w": avg("power_w"), "max_temp_c": max(s["temp_c"] for s in samples),
            "throttle_reasons_seen": sorted({s["throttle"] for s in steady}), "samples": len(samples)}


if __name__ == "__main__":
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "spec": SPEC, "gemm": {}}
    for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("tf32", torch.float32), ("fp8", "fp8")]:
        res["gemm"][name] = {str(n): round(gemm_tflops(n, dt), 1) for n in (4096, 8192, 16384)}
        print(name, res["gemm"][name], flush=True)
    res["hbm"] = hbm_bw(); print("hbm", res["hbm"], flush=True)
    res["pcie"] = pcie_bw(); print("pcie", res["pcie"], flush=True)
    res["sustained_bf16_8192"] = sustained_bf16(60); print("sustained", res["sustained_bf16_8192"], flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", OUT)
