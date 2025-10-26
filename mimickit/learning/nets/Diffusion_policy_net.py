import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

import util.torch_util as torch_util


# --------------------------
# 입력 어댑터: dict -> (N,C,H,W), cond
# --------------------------
class InputAdapter(nn.Module):
    """
    다양한 dict 입력을 (N,C,H,W) 단일 이미지 텐서와
    비-이미지(조건) 벡터 cond (N, D_cond)로 분리해서 반환.
    - 'obs' 키가 있으면 우선 이미지로 사용
    - 없으면 dim>=3 텐서들을 이미지로 간주해 채널 concat
    - dtype=uint8 은 [0,1] float로 변환 (UInt8ToFloat)
    - 서로 다른 크기는 가장 작은 H,W 기준 center-crop
    """
    def __init__(self, sample_inputs: Dict[str, torch.Tensor], non_image_keys_hint: Optional[list] = None):
        super().__init__()
        self.uint8_to_float = torch_util.UInt8ToFloat()

        # 이미지 키 찾기
        if "obs" in sample_inputs:
            self.image_keys = ["obs"]
        else:
            self.image_keys = [k for k, v in sample_inputs.items() if v.dim() >= 3]

        if len(self.image_keys) == 0:
            raise ValueError("No image-like tensor found. Provide 'obs' or at least one tensor with dim>=3.")

        # 조건 키 후보
        self.non_image_keys = [k for k in sample_inputs.keys() if k not in self.image_keys]
        # 힌트가 있으면 우선순위로 포함(중복 제거)
        if non_image_keys_hint:
            for k in non_image_keys_hint:
                if k in sample_inputs and k not in self.non_image_keys and k not in self.image_keys:
                    self.non_image_keys.append(k)

        # target H,W
        shapes = []
        for k in self.image_keys:
            v = sample_inputs[k]
            h, w, c = self._infer_hwc(v)
            shapes.append((h, w))
        self.target_h = min(s[0] for s in shapes)
        self.target_w = min(s[1] for s in shapes)

        # 총 채널 수
        self.total_c = 0
        for k in self.image_keys:
            v = sample_inputs[k]
            _, _, c = self._infer_hwc(v)
            self.total_c += c

        # cond 차원 미리 추정
        self.cond_dim = 0
        for k in self.non_image_keys:
            v = sample_inputs[k]
            # 배치 제외 크기
            shp = v.shape[1:] if v.dim() >= 2 else v.shape
            self.cond_dim += int(np.prod(shp)) if len(shp) > 0 else 1

    @staticmethod
    def _infer_hwc(x: torch.Tensor):
        s = list(x.shape); d = x.dim()
        if d == 3:
            if s[0] <= 16 and s[1] >= 8 and s[2] >= 8:  # (C,H,W)
                C, H, W = s
            else:                                       # (H,W,C)
                H, W, C = s
            return H, W, C
        if d == 4:
            if s[1] <= 16 and s[2] >= 8 and s[3] >= 8:  # (N,C,H,W)
                _, C, H, W = s
                return H, W, C
            elif s[3] <= 16 and s[1] >= 8 and s[2] >= 8: # (N,H,W,C)
                _, H, W, C = s
                return H, W, C
            else:                                        # (T,H,W,C)
                T, H, W, C = s
                return H, W, T * C
        if d == 5:                                       # (N,T,H,W,C)
            _, T, H, W, C = s
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

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        d = x.dim()
        if d == 3:
            if x.shape[0] <= 16 and x.shape[1] >= 8 and x.shape[2] >= 8:  # (C,H,W)
                return x.unsqueeze(0)
            else:                                                         # (H,W,C)
                return x.permute(2, 0, 1).unsqueeze(0)
        if d == 4:
            if x.shape[1] <= 16 and x.shape[2] >= 8 and x.shape[3] >= 8:  # (N,C,H,W)
                return x
            elif x.shape[3] <= 16 and x.shape[1] >= 8 and x.shape[2] >= 8: # (N,H,W,C)
                return x.permute(0, 3, 1, 2)
            else:                                                          # (T,H,W,C) -> (1, T*C, H, W)
                T, H, W, C = x.shape
                return x.reshape(1, T * C, H, W)
        if d == 5:                                                         # (N,T,H,W,C) -> (N, T*C, H, W)
            N, T, H, W, C = x.shape
            return x.permute(0, 1, 4, 2, 3).reshape(N, T * C, H, W)
        raise ValueError(f"Unsupported tensor rank for image: {x.shape}")

    def forward(self, x: Dict[str, torch.Tensor]):
        imgs = []
        for k in self.image_keys:
            t = x[k]
            if t.dtype == torch.uint8:
                t = self.uint8_to_float(t)
            t = self._to_nchw(t)
            t = self._center_crop(t, self.target_h, self.target_w)
            imgs.append(t)
        obs = torch.cat(imgs, dim=1) if len(imgs) > 1 else imgs[0]  # (N,C,H,W)

        cond_list = []
        for k in self.non_image_keys:
            t = x[k]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            cond_list.append(t.reshape(t.shape[0], -1))
        cond = torch.cat(cond_list, dim=1) if len(cond_list) > 0 else torch.zeros(obs.shape[0], 0, device=obs.device)
        return obs, cond  # (N,C,H,W), (N,D_cond)


# --------------------------
# U-Net 백본
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
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = nn.functional.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class SmallUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, activation):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, activation)
        self.down1 = Down(base_ch, base_ch*2, activation)
        self.down2 = Down(base_ch*2, base_ch*4, activation)
        self.up1   = Up(base_ch*4 + base_ch*2, base_ch*2, activation)
        self.up2   = Up(base_ch*2 + base_ch, base_ch, activation)
        self.out_conv = nn.Conv2d(base_ch, base_ch, 1)

    def forward(self, x):  # (N,C,H,W)
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        b  = self.down2(e2)
        u1 = self.up1(b, e2)
        u2 = self.up2(u1, e1)
        f  = self.out_conv(u2)                 # (N, base_ch, H, W)
        g  = f.mean(dim=(-2, -1))              # GAP -> (N, base_ch)
        return g


# --------------------------
# Diffusion 모듈
# --------------------------
class SinusoidalPosEmb(nn.Module):
    """DDPM/DiT 류에서 쓰는 t 임베딩."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (N,) float or int
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=timesteps.dtype) *
            -(np.log(10000.0) / (half - 1 + 1e-8))
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb  # (N, dim)

class DiffusionHead(nn.Module):
    """
    U-Net 특징 + 조건(cond) + t 임베딩 -> ε 예측
    출력 차원 out_dim은 action/target shape에 맞춰 결정.
    """
    def __init__(self, in_feat: int, cond_dim: int, t_dim: int, out_dim: int, activation):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat + cond_dim + t_dim, 512),
            activation(),
            nn.Linear(512, 512),
            activation(),
            nn.Linear(512, out_dim),
        )
        # 원 코드 스타일: bias zero init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, feat, cond, t_emb):
        x = torch.cat([feat, cond, t_emb], dim=1)
        return self.mlp(x)  # (N, out_dim) -> ε_pred


class UNetDiffusionPolicy(nn.Module):
    """
    Forward(x_dict, t) -> epsilon_pred
    - x_dict: 관측/조건 dict
    - t: 타임스텝 텐서 (N,)
    """
    def __init__(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        activation,
        base_ch: int = 64,
        t_emb_dim: int = 128,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        # 어떤 키를 보고 out_dim을 정할지 추론
        self.action_key = None
        if "action" in sample_inputs:
            self.action_key = "action"
        elif "target" in sample_inputs:
            self.action_key = "target"

        # 입력 어댑터
        # action/target은 cond로 쓰되, out_dim 추정에만 사용
        non_img_hint = [k for k in ["action", "target"] if k in sample_inputs]
        self.adapter = InputAdapter(sample_inputs, non_image_keys_hint=non_img_hint)

        # 샘플 통과로 in_ch 파악
        with torch.no_grad():
            sample = {k: (v if v.dim() >= 3 else v.unsqueeze(0)) for k, v in sample_inputs.items()}
            obs, cond = self.adapter(sample)  # (N,C,H,W), (N,D_cond)
        in_ch = obs.shape[1]
        cond_dim = cond.shape[1]

        # U-Net 백본
        self.unet = SmallUNet(in_ch=in_ch, base_ch=base_ch, activation=activation)

        # t 임베딩
        self.t_emb = SinusoidalPosEmb(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            activation(),
            nn.Linear(t_emb_dim, t_emb_dim),
            activation(),
        )
        for m in self.t_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        # 출력 차원 결정
        if out_dim is None:
            if self.action_key is not None:
                v = sample_inputs[self.action_key]
                shp = v.shape[1:] if v.dim() >= 2 else v.shape
                out_dim = int(np.prod(shp)) if len(shp) > 0 else 1
            else:
                out_dim = 64  # 기본값

        # 헤드
        self.head = DiffusionHead(in_feat=base_ch, cond_dim=cond_dim, t_dim=t_emb_dim, out_dim=out_dim, activation=activation)
        self.out_dim = out_dim
        self.base_ch = base_ch

    def forward(self, x: Dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        obs, cond = self.adapter(x)           # (N,C,H,W), (N,D_cond)
        feat = self.unet(obs)                 # (N, base_ch)
        t_emb = self.t_proj(self.t_emb(t))    # (N, t_emb_dim)
        eps = self.head(feat, cond, t_emb)    # (N, out_dim)
        return eps


# --------------------------
# build_net: 기존 포맷 유지
# --------------------------
def build_net(input_dict: Dict[str, torch.Tensor], activation) -> Tuple[nn.Module, dict]:
    """
    Diffusion Policy Network with U-Net backbone
    - 시그니처/반환 형식 유지: (net, info)
    - forward(net): net(x_dict, t) 형태로 사용하세요.
    """
    base_ch  = 64
    t_emb_dim = 128

    net = UNetDiffusionPolicy(
        sample_inputs=input_dict,
        activation=activation,
        base_ch=base_ch,
        t_emb_dim=t_emb_dim,
        out_dim=None,  # auto-detect from 'action' or 'target' if present, else 64
    )

    info = dict(
        type="unet_diffusion_policy",
        expects_forward="net(x_dict: dict[str, Tensor], t: Tensor[N]) -> eps_pred[N, out_dim]",
        input_adapter=dict(
            image_keys=[k for k in (["obs"] if "obs" in input_dict else input_dict.keys()) if input_dict[k].dim() >= 3],
            non_image_keys=[k for k in input_dict.keys() if not (input_dict[k].dim() >= 3 and (k == "obs" or "obs" not in input_dict))],
            total_in_channels=getattr(net.adapter, "total_c", None),
            target_hw=(getattr(net.adapter, "target_h", None), getattr(net.adapter, "target_w", None)),
            cond_dim=getattr(net.adapter, "cond_dim", None),
        ),
        unet_base_ch=base_ch,
        t_embedding_dim=t_emb_dim,
        out_dim=net.out_dim,
        epsilon_target="DDPM epsilon prediction",
        init_note="All Linear biases zero-initialized",
    )
    return net, info
