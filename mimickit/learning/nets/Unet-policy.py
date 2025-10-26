import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

import util.torch_util as torch_util

# --------------------------
# 입력 어댑터: dict -> (N,C,H,W)
# --------------------------
class InputAdapter(nn.Module):
    """
    다양한 dict 입력을 (N,C,H,W) 단일 텐서로 통합해 반환.
    규칙:
      1) 'obs'가 있으면 최우선 사용
      2) 없으면 dim>=3인 텐서들을 이미지로 간주해 C축으로 concat
      3) (H,W,C)/(N,H,W,C)/(T,H,W,C)/(N,T,H,W,C)/CHW/NCHW 등을 NCHW로 변환
      4) dtype=uint8이면 UInt8ToFloat 사용
      5) 서로 다른 크기는 가장 작은 H,W로 center-crop
    """
    def __init__(self, sample_inputs: Dict[str, torch.Tensor]):
        super().__init__()
        self.uint8_to_float = torch_util.UInt8ToFloat()

        if "obs" in sample_inputs:
            self.image_keys = ["obs"]
        else:
            self.image_keys = [k for k, v in sample_inputs.items() if v.dim() >= 3]
            if len(self.image_keys) == 0:
                raise ValueError("No image-like tensor found. Provide 'obs' or at least one tensor with dim>=3.")

        # target H,W는 가장 작은 크기로 고정
        shapes = []
        for k in self.image_keys:
            v = sample_inputs[k]
            h, w, c = self._infer_hwc(v)
            shapes.append((h, w))
        self.target_h = min(s[0] for s in shapes)
        self.target_w = min(s[1] for s in shapes)

        # 총 채널 수 추정
        self.total_c = 0
        for k in self.image_keys:
            v = sample_inputs[k]
            _, _, c = self._infer_hwc(v)
            self.total_c += c

    @staticmethod
    def _infer_hwc(x: torch.Tensor):
        dims = x.dim()
        s = list(x.shape)
        if dims == 3:
            # (C,H,W) or (H,W,C)
            if s[0] <= 16 and s[1] >= 8 and s[2] >= 8:
                C, H, W = s
            else:
                H, W, C = s
            return H, W, C
        if dims == 4:
            # (N,C,H,W) or (N,H,W,C) or (T,H,W,C)
            if s[1] <= 16 and s[2] >= 8 and s[3] >= 8:
                N, C, H, W = s
                return H, W, C
            elif s[3] <= 16 and s[1] >= 8 and s[2] >= 8:
                N, H, W, C = s
                return H, W, C
            else:
                T, H, W, C = s
                return H, W, T * C
        if dims == 5:
            # (N,T,H,W,C)
            N, T, H, W, C = s
            return H, W, T * C
        raise ValueError(f"Unsupported tensor rank for image: {x.shape}")

    @staticmethod
    def _center_crop(x: torch.Tensor, target_h: int, target_w: int):
        h, w = x.shape[-2], x.shape[-1]
        if h == target_h and w == target_w:
            return x
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return x[..., top:top+target_h, left:left+target_w]

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.dim()
        if dims == 3:
            # (C,H,W) or (H,W,C)
            if x.shape[0] <= 16 and x.shape[1] >= 8 and x.shape[2] >= 8:
                x = x.unsqueeze(0)  # (1,C,H,W)
            else:
                x = x.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
            return x
        if dims == 4:
            if x.shape[1] <= 16 and x.shape[2] >= 8 and x.shape[3] >= 8:
                return x  # (N,C,H,W)
            elif x.shape[3] <= 16 and x.shape[1] >= 8 and x.shape[2] >= 8:
                return x.permute(0, 3, 1, 2)  # (N,H,W,C)->(N,C,H,W)
            else:
                # (T,H,W,C) -> (1, T*C, H, W)
                T, H, W, C = x.shape
                x = x.reshape(1, T * C, H, W)
                return x
        if dims == 5:
            # (N,T,H,W,C) -> (N, T*C, H, W)
            N, T, H, W, C = x.shape
            return x.permute(0, 1, 4, 2, 3).reshape(N, T * C, H, W)
        raise ValueError(f"Unsupported tensor rank for image: {x.shape}")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = []
        for k in self.image_keys:
            t = x[k]
            if t.dtype == torch.uint8:
                t = self.uint8_to_float(t)
            t = self._to_nchw(t)
            t = self._center_crop(t, self.target_h, self.target_w)
            imgs.append(t)
        obs = torch.cat(imgs, dim=1) if len(imgs) > 1 else imgs[0]
        return obs  # (N,C,H,W)


# --------------------------
# U-Net 블록
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, activation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            activation(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            activation(),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, cin, cout, activation):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(cin, cout, activation)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, cout, activation):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(cin, cout, activation)
    def forward(self, x, skip):
        x = self.up(x)
        # 패딩으로 크기 맞추기
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = nn.functional.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class SmallUNet(nn.Module):
    """
    입력: (N,C,H,W)
    출력: GAP로 요약된 feature (N, base_ch)
    """
    def __init__(self, in_ch: int, base_ch: int, activation):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, activation)
        self.down1 = Down(base_ch, base_ch*2, activation)
        self.down2 = Down(base_ch*2, base_ch*4, activation)
        self.up1   = Up(base_ch*4 + base_ch*2, base_ch*2, activation)
        self.up2   = Up(base_ch*2 + base_ch, base_ch, activation)
        self.out_conv = nn.Conv2d(base_ch, base_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        b  = self.down2(e2)
        u1 = self.up1(b, e2)
        u2 = self.up2(u1, e1)
        f  = self.out_conv(u2)        # (N, base_ch, H, W)
        g  = f.mean(dim=(-2, -1))     # GAP -> (N, base_ch)
        return g


# --------------------------
# U-Net Policy Network
# --------------------------
class UNetPolicy(nn.Module):
    """
    Dict 입력을 받아:
      InputAdapter -> U-Net -> MLP(head) -> policy embedding
    """
    def __init__(self, sample_inputs: Dict[str, torch.Tensor], activation, fc_sizes=(512,), base_ch: int = 64):
        super().__init__()
        self.adapter = InputAdapter(sample_inputs)

        # 어댑터 샘플 통과로 in_ch 파악
        with torch.no_grad():
            sample = {k: (v if v.dim() >= 3 else v.unsqueeze(0)) for k, v in sample_inputs.items()}
            obs_std = self.adapter(sample)  # (N,C,H,W)
        in_ch = obs_std.shape[1]

        # U-Net 백본
        self.unet = SmallUNet(in_ch=in_ch, base_ch=base_ch, activation=activation)

        # Head: [base_ch] -> fc_sizes
        head_layers = []
        in_feat = base_ch
        for out_size in fc_sizes:
            lin = nn.Linear(in_feat, out_size)
            nn.init.zeros_(lin.bias)
            head_layers += [lin, activation()]
            in_feat = out_size
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.adapter(x)   # (N,C,H,W), float in [0,1]
        z   = self.unet(obs)    # (N, base_ch)
        y   = self.head(z)      # (N, fc_sizes[-1])
        return y


def build_net(input_dict: Dict[str, torch.Tensor], activation) -> Tuple[nn.Module, dict]:
    """
    형식 유지:
      - 시그니처: (input_dict, activation)
      - 반환: (net, info)
    CNN policy를 U-Net 백본으로 대체하고, 입력은 dict를 유연하게 수용.
    출력 차원은 fc_sizes[-1] (기본 512)로 기존 정책 헤드 규격을 맞춥니다.
    """
    fc_sizes = [512]        # 기존 코드의 policy head 크기
    base_ch  = 64           # U-Net base channel (필요시 32~128로 조절 가능)

    net = UNetPolicy(input_dict, activation, fc_sizes=tuple(fc_sizes), base_ch=base_ch)

    info = dict(
        type="unet_policy",
        input_adapter="auto_dict_to_NCHW_concat",
        image_keys=[k for k in (["obs"] if "obs" in input_dict else input_dict.keys()) if input_dict[k].dim() >= 3],
        total_in_channels=getattr(net.adapter, "total_c", None),
        target_hw=(getattr(net.adapter, "target_h", None), getattr(net.adapter, "target_w", None)),
        unet_base_ch=base_ch,
        mlp_head=fc_sizes,
        output_dim=fc_sizes[-1],
    )

    return net, info
