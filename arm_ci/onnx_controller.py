"""
onnx_controller.py —— 把推理脚本的模型换成 ONNX Runtime

设计要点：**不改原脚本任何控制逻辑**。
    原脚本 DroneInferenceController.__init__ 里做的是:
        self.model = self.load_model(model_path)
        self.model.eval()
    而 predict_control() 里是:
        joystick_command_normalized = self.model(image_tensor)
    所以我们只需要让 load_model() 返回一个"看起来像 PyTorch 模型"的包装对象，
    有 eval() / to() / __call__ 三个方法就够了。其余代码一行不用动。

用法：
    from onnx_controller import OnnxDroneInferenceController
    ctrl = OnnxDroneInferenceController(model_path, norm_path)
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


# ---------------------------------------------------------------------------
# ONNX 导出兼容层（和 export_onnx.py 里的实现一致）
#
# 原模型的 AdaptiveAvgPool2d((3,4)) 作用在 7x10 上、((2,2)) 作用在 3x4 上，
# 输出尺寸都不是输入的整数因子，ONNX 不支持这种 adaptive_avg_pool2d，导出会报错。
# 这里用切片 + mean 复刻 PyTorch 的精确窗口规则，可导出且数值等价（float64 误差 2.2e-16）。
# ---------------------------------------------------------------------------
class ExportableAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(output_size) if isinstance(output_size, (tuple, list)) \
            else (output_size, output_size)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        oh, ow = self.output_size
        hs = [(i * H // oh, math.ceil((i + 1) * H / oh)) for i in range(oh)]
        ws = [(j * W // ow, math.ceil((j + 1) * W / ow)) for j in range(ow)]
        rows = [[x[..., h0:h1, w0:w1].mean(dim=(-2, -1)) for (w0, w1) in ws]
                for (h0, h1) in hs]
        return torch.stack([torch.stack(r, -1) for r in rows], -2)


def make_exportable(model):
    """把模型里所有 AdaptiveAvgPool2d 换成可导出的等价实现（原地修改，返回替换个数）"""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            out = mod.output_size
            out = (out, out) if isinstance(out, int) else tuple(out)
            parent = model
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            setattr(parent, name.split(".")[-1], ExportableAdaptiveAvgPool2d(out))
            n += 1
    return n


def export_model_to_onnx(model, out_path, seq_len, image_size, opset=17, verbose=True):
    """把 PyTorch 模型导出成 .onnx（会自动处理 AdaptiveAvgPool2d）"""
    if verbose:
        print(f"  导出 ONNX -> {out_path}")
    n = make_exportable(model)
    if verbose:
        print(f"    AdaptiveAvgPool2d 替换: {n} 处")
    model = model.eval()
    dummy = torch.rand(1, seq_len, 3, *image_size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["images"], output_names=["command"],
        opset_version=opset, do_constant_folding=True, dynamo=False)
    if verbose:
        print(f"    完成，{os.path.getsize(out_path)/1e6:.1f} MB")
    return out_path


class OnnxModelWrapper:
    """
    伪装成 PyTorch 模型的 ONNX 推理包装。
    对外暴露 eval() / to() / __call__，让原脚本的控制逻辑无需修改。
    """

    def __init__(self, onnx_path, threads=0, verbose=True):
        if ort is None:
            raise ImportError("需要 onnxruntime: pip install onnxruntime")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or min(4, os.cpu_count() or 4)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.onnx_path = onnx_path
        if verbose:
            print(f"  ONNX Runtime 已加载: {onnx_path}")
            print(f"    输入 {tuple(self.sess.get_inputs()[0].shape)}  "
                  f"线程 {so.intra_op_num_threads}")

    # 让原脚本的 model.eval() / model.to(device) 调用不出错
    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def __call__(self, images):
        """输入 torch 张量，返回 torch 张量（和 PyTorch 模型行为一致）"""
        arr = images.detach().cpu().numpy().astype(np.float32)
        out = self.sess.run(None, {self.input_name: arr})[0]
        return torch.from_numpy(np.ascontiguousarray(out))


def build_onnx_drone_controller(base_cls):
    """
    基于原脚本的 DroneInferenceController，生成一个用 ONNX 推理的子类。
    只覆盖 load_model()，其它全部继承。
    """

    class OnnxDroneInferenceController(base_cls):
        onnx_threads = 0

        def load_model(self, model_path):
            onnx_path = os.path.splitext(model_path)[0] + ".onnx"

            # 先读 checkpoint 拿配置
            ckpt = self._read_checkpoint(model_path)
            cfg = ckpt.get("config", {})
            self.seq_len = int(cfg.get("sequence_length", 5))
            self.img_size = tuple(cfg.get("image_size", (240, 320)))

            if not os.path.exists(onnx_path):
                print(f"  没找到 {onnx_path}，现在导出...")
                net = self._build_torch_model(cfg)
                net.load_state_dict(ckpt["model_state_dict"])
                export_model_to_onnx(net, onnx_path, self.seq_len, self.img_size)
            else:
                print(f"  复用已导出的 {onnx_path}")

            return OnnxModelWrapper(onnx_path, threads=self.onnx_threads)

        # ---- 下面是辅助方法，名字带下划线，不会被父类逻辑误用 ----
        def _read_checkpoint(self, model_path):
            return torch.load(model_path, map_location="cpu", weights_only=False)

        def _build_torch_model(self, cfg):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_slim_for_onnx",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "slim", "slim_model.py"))
            # 默认用原版结构；如果 checkpoint 标了 arch=slim 则用轻量版
            if cfg.get("arch") == "slim":
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.LSTMSpikeDroneModelSlim(
                    sequence_length=int(cfg.get("sequence_length", 5)),
                    joystick_command_dim=int(cfg.get("joystick_command_dim", 4)),
                    image_size=tuple(cfg.get("image_size", (240, 320))),
                    use_spike=bool(cfg.get("use_spike", True)),
                    feat_channels=int(cfg.get("feat_channels", 128)),
                    lstm_hidden=int(cfg.get("lstm_hidden", 256)),
                    lstm_layers=int(cfg.get("lstm_layers", 1)),
                    bidirectional=bool(cfg.get("bidirectional", False)),
                    head_dim=int(cfg.get("head_dim", 256)))
            # 原版：由调用方注入模型类
            if not hasattr(self, "_base_model_cls") or self._base_model_cls is None:
                raise RuntimeError("原版结构需要先设置 self._base_model_cls")
            return self._base_model_cls(
                sequence_length=int(cfg.get("sequence_length", 5)),
                joystick_command_dim=int(cfg.get("joystick_command_dim", 4)),
                image_size=tuple(cfg.get("image_size", (240, 320))),
                use_spike=bool(cfg.get("use_spike", True)))

    return OnnxDroneInferenceController
