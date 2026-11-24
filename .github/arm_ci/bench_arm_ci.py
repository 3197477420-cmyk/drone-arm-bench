#!/usr/bin/env python3
"""
bench_arm_ci.py —— 在真实 ARM64 硬件上测这个模型的推理耗时（不需要模型权重文件）

为什么不需要 checkpoint：
    推理耗时由**网络结构**决定，和权重数值无关。所以用随机初始化的同结构模型，
    测出来的时间完全一样。这样就避开了上传 100MB 模型的问题。

依赖：torch、snnorch（可选 onnxruntime）

用法：
    python3 bench_arm_ci.py
    python3 bench_arm_ci.py --threads 4 --repeats 10 --skip-onnx
"""
import argparse
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="线程数，可传多个")
ap.add_argument("--repeats", type=int, default=8)
ap.add_argument("--skip-onnx", action="store_true")
a = ap.parse_args()

# 原版的 _init_weights 会对 2048x2048 的 LSTM 矩阵做 orthogonal_（内部 QR 分解），
# 在慢机器上要几十秒。耗时与权重数值无关，换成廉价初始化。
torch.nn.init.orthogonal_ = lambda t, gain=1.0: torch.nn.init.xavier_uniform_(t)

from slim_model import LSTMSpikeDroneModelSlim  # noqa: E402


def sysinfo():
    print("=" * 78)
    print("运行环境")
    print("=" * 78)
    print(f"  架构        : {platform.machine()}")
    print(f"  内核        : {platform.release()}")
    print(f"  发行版      : {platform.platform()}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  torch       : {torch.__version__}")
    print(f"  CPU 核数    : {os.cpu_count()}")
    try:
        with open("/proc/meminfo") as f:
            mem = [l for l in f if l.startswith("MemTotal")][0]
        print(f"  内存        : {int(mem.split()[1])/1024/1024:.1f} GB")
    except Exception:
        pass
    # ARM 上通常在 /proc/cpuinfo 的 "CPU part" / "model name" 里
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        for key in ("model name", "Hardware", "CPU implementer", "CPU part", "CPU variant"):
            for line in txt.splitlines():
                if line.lower().startswith(key.lower()):
                    print(f"  {key:<12}: {line.split(':', 1)[1].strip()}")
                    break
    except Exception:
        pass
    try:
        l1, l5, l15 = os.getloadavg()
        print(f"  负载(1/5/15): {l1:.2f} / {l5:.2f} / {l15:.2f}")
        if l1 > (os.cpu_count() or 1) * 0.5:
            print("  ⚠️  机器有其它负载，数字会偏慢（CI 共享机器上常见，已尽量用中位数）")
    except Exception:
        pass
    print()


def bench(model, hw, repeats):
    x = torch.rand(1, 5, 3, *hw)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(x)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000, min(ts) * 1000


# 用 slim_model 的参数化能力复现两种结构：
#   feat_channels=512, lstm_hidden=512, layers=3, bidirectional  → 原版 24.97M
#   feat_channels=128, lstm_hidden=256, layers=1, 单向           → 轻量版 1.58M
CASES = [
    ("原版 24.97M  240x320", dict(feat_channels=512, lstm_hidden=512, lstm_layers=3,
                                  bidirectional=True), (240, 320)),
    ("轻量 1.58M  240x320", dict(feat_channels=128, lstm_hidden=256, lstm_layers=1,
                                 bidirectional=False), (240, 320)),
    ("轻量 1.58M  160x213", dict(feat_channels=128, lstm_hidden=256, lstm_layers=1,
                                 bidirectional=False), (160, 213)),
    ("轻量 1.58M  120x160", dict(feat_channels=128, lstm_hidden=256, lstm_layers=1,
                                 bidirectional=False), (120, 160)),
]

sysinfo()
print("=" * 78)
print(f"推理耗时（中位数 of {a.repeats}，随机权重，结构与真实模型一致）")
print("=" * 78)

rows = []
for nt in a.threads:
    torch.set_num_threads(nt)
    print(f"\n  --- torch 线程数 = {nt} ---")
    print(f"  {'配置':<24} {'参数量':>9} {'中位':>9} {'最快':>9} {'FPS':>8}")
    print(f"  {'-'*24} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
    for name, kw, hw in CASES:
        m = LSTMSpikeDroneModelSlim(sequence_length=5, head_dim=256, **kw)
        n_par = m.n_params()
        med, mn = bench(m, hw, a.repeats if n_par < 20e6 else max(4, a.repeats // 2))
        rows.append((nt, name, n_par, med))
        print(f"  {name:<24} {n_par/1e6:>8.2f}M {med:>8.1f}ms {mn:>8.1f}ms {1000/med:>7.2f}")

# ---------------------------------------------------------------- ONNX
if not a.skip_onnx:
    try:
        import onnxruntime as ort
        import onnx  # noqa: F401
        sys.path.insert(0, HERE)
        from onnx_controller import export_model_to_onnx, OnnxModelWrapper

        print()
        print("=" * 78)
        print(f"ONNX Runtime（{ort.__version__}）")
        print("=" * 78)
        best_nt = a.threads[-1]
        torch.set_num_threads(best_nt)
        outdir = "/tmp/onnx_ci"
        os.makedirs(outdir, exist_ok=True)
        print(f"  torch 线程数 = {best_nt}")
        print(f"  {'配置':<24} {'PyTorch':>10} {'ONNX':>10} {'加速':>8} {'FPS':>8}")
        print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name, kw, hw in CASES:
            m = LSTMSpikeDroneModelSlim(sequence_length=5, head_dim=256, **kw)
            p = os.path.join(outdir, name.replace(" ", "_").replace("x", "x") + ".onnx")
            export_model_to_onnx(m, p, 5, hw, verbose=False)
            w = OnnxModelWrapper(p, threads=best_nt, verbose=False)
            x = torch.rand(1, 5, 3, *hw)
            with torch.no_grad():
                for _ in range(2):
                    m(x)
                t_pt = statistics.median(
                    [(lambda t0: (m(x), time.perf_counter() - t0)[1])(time.perf_counter())
                     for _ in range(max(4, a.repeats // 2))]) * 1000
            for _ in range(2):
                w(x)
            t_on = statistics.median(
                [(lambda t0: (w(x), time.perf_counter() - t0)[1])(time.perf_counter())
                 for _ in range(a.repeats)]) * 1000
            print(f"  {name:<24} {t_pt:>9.1f}ms {t_on:>9.1f}ms {t_pt/t_on:>7.2f}x {1000/t_on:>7.2f}")
    except ImportError:
        print("\n  （未安装 onnxruntime，跳过 ONNX 部分）")
        print("   安装: pip install \"numpy<2\" onnx onnxruntime")

# ---------------------------------------------------------------- 结论
print()
print("=" * 78)
print("结论")
print("=" * 78)
print("  控制周期参考: 30 ms（timestep=0.03s）→ 实时需要 33.3 FPS")
print("  可用性门槛（来自降频实验）: 16.7 Hz 几乎无损 / 5.6 Hz 可接受 / 2.8 Hz 开始漂")
print()
base_fps = [r for r in rows if r[1].startswith("原版") and r[0] == a.threads[-1]]
if base_fps:
    print(f"  原版在你这台 ARM 机器上: {1000/base_fps[0][3]:.2f} FPS ({base_fps[0][3]:.0f} ms)")
best = min([r for r in rows if r[0] == a.threads[-1]], key=lambda r: r[3])
print(f"  最快的结构: {best[1]} → {1000/best[3]:.2f} FPS ({best[3]:.0f} ms)")
print()
print("  把这份输出发回来，就能确定树莓派 5 上到底能跑多少 Hz。")
