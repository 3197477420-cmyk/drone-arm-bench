"""
slim_model.py —— 轻量化版 LSTMSpikeDroneModel（可配置，drop-in 替换）

和原版（inference_sine_LSTMSNN_v6.py 里的 LSTMSpikeDroneModel）的区别：

    原版                              轻量版
    ────────────────────────────────  ──────────────────────────────────────
    feature_compress: 256→512 通道     256→feat_channels（默认 128）
    flatten 后 2048 维                 128*2*2 = 512 维
    LSTM: 输入2048, 隐藏512            输入512, 隐藏256（默认可调）
          3 层双向                    1 层单向
    lstm_output_layer: 1024→256        256→256
    总参数 ~24.97 M                    ~3.3 M（默认配置）

设计依据（见 README_PI5.md）：
  1. 原版 LSTM 占 92.5% 参数、35% 耗时；CNN 占 6.3% 参数、65% 耗时
  2. 最大的冗余是「3×4 特征图上开 512 通道然后 flatten 成 2048」这一步
  3. 双向 LSTM 在实时控制里取最后时间步输出，反向只看得到 1 帧，收益极低

注意：**这不是 drop-in 等价替换**。结构变了，必须重新训练。
      convert_warmstart.py 会做权重迁移作为热启动，但精度需要重训恢复。

用法：
    from slim_model import LSTMSpikeDroneModelSlim
    model = LSTMSpikeDroneModelSlim(sequence_length=5)
    y = model(torch.rand(1, 5, 3, 240, 320))
"""
import torch
import torch.nn as nn

try:
    import snntorch as snn
except ImportError:  # pragma: no cover
    snn = None


class SpikingNeuronLayer(nn.Module):
    """脉冲神经元层，输出膜电位（连续），与原版一致"""

    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        if snn is None:
            raise ImportError("需要 snntorch：pip install snntorch")
        self.neuron = snn.Leaky(beta=decay, threshold=threshold,
                                reset_mechanism="zero", output=False)

    def forward(self, x, mem=None):
        if mem is None:
            self.neuron.reset_mem()
        _, mem = self.neuron(x, mem)
        return mem


class SpikeActivation(nn.Module):
    def __init__(self, threshold=0.2, decay=0.85):
        super().__init__()
        self.spike_layer = SpikingNeuronLayer(threshold=threshold, decay=decay)

    def forward(self, x):
        return self.spike_layer(x)


class SpikeCNNBlock(nn.Module):
    """与原版完全一致（这部分没有冗余，保留）"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spike=True):
        super().__init__()
        self.use_spike = use_spike
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.spike_layer = SpikingNeuronLayer(threshold=0.5, decay=0.8) if use_spike else None
        self.activation = nn.ReLU() if not use_spike else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.spike_layer(x) if self.use_spike else self.activation(x)
        return self.pool(x)


class SlimVisionEncoder(nn.Module):
    """视觉编码器：前 4 层与原版相同，只把最后的特征压缩从 512 通道降到 feat_channels"""

    def __init__(self, feat_channels=128, use_spike=True, image_size=(240, 320)):
        super().__init__()
        self.feat_channels = feat_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.spike_block1 = SpikeCNNBlock(32, 64, use_spike=use_spike)
        self.spike_block2 = SpikeCNNBlock(64, 128, use_spike=use_spike)
        self.spike_block3 = SpikeCNNBlock(128, 256, use_spike=use_spike)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 4))
        # ↓ 这里从 512 降到 feat_channels
        self.feature_compress = nn.Sequential(
            nn.Conv2d(256, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())

    @property
    def out_dim(self):
        return self.feat_channels * 2 * 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.spike_block1(x)
        x = self.spike_block2(x)
        x = self.spike_block3(x)
        x = self.adaptive_pool(x)
        return self.feature_compress(x)


class LSTMSpikeDroneModelSlim(nn.Module):
    """
    轻量版。forward(images) 的调用方式和原版完全一致：
        images: (B, T, 3, H, W)  ->  (B, joystick_command_dim)
    """

    def __init__(self, sequence_length=5, joystick_command_dim=4, image_size=(240, 320),
                 use_spike=True, feat_channels=128, lstm_hidden=256,
                 lstm_layers=1, bidirectional=False, head_dim=256):
        super().__init__()
        self.sequence_length = sequence_length
        self.use_spike = use_spike
        self.feat_channels = feat_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.head_dim = head_dim

        self.vision_encoder = SlimVisionEncoder(feat_channels, use_spike, image_size)
        feat_dim = self.vision_encoder.out_dim

        self.vision_lstm = nn.LSTM(
            feat_dim, lstm_hidden,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=0.3 if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional)

        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        lay = [nn.Linear(lstm_out, head_dim), nn.LayerNorm(head_dim)]
        lay.append(SpikeActivation(threshold=0.2, decay=0.85) if use_spike else nn.ReLU())
        lay.append(nn.Dropout(0.3))
        self.lstm_output_layer = nn.Sequential(*lay)

        tmp = [nn.Linear(head_dim, 128), nn.LayerNorm(128)]
        tmp.append(SpikeActivation(threshold=0.2, decay=0.85) if use_spike else nn.ReLU())
        tmp.extend([nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)])
        self.temporal_processor = nn.Sequential(*tmp)

        self.joystick_command_predictor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, joystick_command_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, images):
        b, t = images.shape[0], images.shape[1]
        flat = images.reshape(b * t, *images.shape[2:])
        feat = self.vision_encoder(flat).reshape(b, t, -1)
        out, _ = self.vision_lstm(feat)
        x = self.lstm_output_layer(out[:, -1, :])
        x = self.temporal_processor(x)
        return self.joystick_command_predictor(x)

    # ------------------------------------------------------------------
    def config(self):
        """保存 checkpoint 用的配置"""
        return {
            "sequence_length": self.sequence_length,
            "joystick_command_dim": int(self.joystick_command_predictor[-1].out_features),
            "image_size": (240, 320),
            "use_spike": self.use_spike,
            "feat_channels": self.feat_channels,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "bidirectional": self.bidirectional,
            "head_dim": self.head_dim,
            "arch": "slim",
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
