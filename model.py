import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean(dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w


class Physics3DLayer(nn.Module):
    DIPOLE_INIT_POSITIONS_CM = {
        1: [ [1.0, -1.5, 1.5] ],
        2: [ [1.0, -1.5, 1.5], [-1.5, -1.0, 0.5] ],
        3: [ [1.0, -1.5, 1.5], [-1.5, -1.0, 0.5], [2.5, -2.0, -1.0] ],
        4: [ [1.0, -1.5, 1.5], [-1.5, -1.0, 0.5], [2.5, -2.0, -1.0], [0.0, -3.0, 0.5] ],
    }

    def __init__(self, num_dipoles=3):
        super().__init__()
        self.num_dipoles = num_dipoles

        electrodes_3d = {
            'RA': [-15.0, 15.0, 0], 'LA': [15.0, 15.0, 0], 'LL': [10.0, -25.0, 0],
            'V1': [-2.0, 0.0, 10.0],  'V2': [2.0, 0.0, 10.0],  'V3': [4.0, -2.0, 9.0],
            'V4': [6.0, -4.0, 8.0],   'V5': [10.0, -4.0, 4.0], 'V6': [12.0, -4.0, 0.0],
        }

        self.register_buffer('RA_base', torch.tensor(electrodes_3d['RA'], dtype=torch.float32))
        self.register_buffer('LA_base', torch.tensor(electrodes_3d['LA'], dtype=torch.float32))
        self.register_buffer('LL_base', torch.tensor(electrodes_3d['LL'], dtype=torch.float32))

        chest_keys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.register_buffer('chest_base', torch.tensor([electrodes_3d[k] for k in chest_keys], dtype=torch.float32))

        # 【改 1】电极偏移与偶极子位置改为可学习参数。
        # 原实现用 register_buffer → 不接收梯度 → repel_loss 与位置学习全部无效，
        # 物理头退化成固定线性投影，这是"加偶极子无增益"的根因之一。
        self.limb_offset  = nn.Parameter(torch.zeros(3, 3))
        self.chest_offset = nn.Parameter(torch.zeros(6, 3))

        self.lead_gain = nn.Parameter(torch.ones(8))

        if num_dipoles not in self.DIPOLE_INIT_POSITIONS_CM:
            supported = sorted(self.DIPOLE_INIT_POSITIONS_CM.keys())
            raise ValueError(f"num_dipoles={num_dipoles} 暂无预置位置, 当前支持: {supported}。")
        init_positions = torch.tensor(self.DIPOLE_INIT_POSITIONS_CM[num_dipoles], dtype=torch.float32)
        self.dipole_pos = nn.Parameter(init_positions.clone())

        # 【改 3b】可学习衰减指数 p (初始 3.0 = point dipole far-field)。
        # 胸前电极离心脏 ~10cm, 处于近-中场, 真实有效衰减通常在 [2.0, 2.5] 而非 3.0。
        # 训练中 clamp 到 [1.5, 4.5] 防数值病态。
        self.decay_exponent = nn.Parameter(torch.tensor(3.0))

        # 【改 3c】可学习 3x3 各向异性导电张量 (初始单位阵 = 当前各向同性行为)。
        # 真实躯干肺/肌肉/血液导电各向异性, 沿不同坐标轴电流扩散速度差异可达 3:1。
        self.conductivity = nn.Parameter(torch.eye(3))

    def _transfer(self, elec_pos):
        diff = elec_pos.unsqueeze(0) - self.dipole_pos              # (D, 3)
        # 各向异性导电加权：等价于让"等效电流方向"可以被旋转/缩放
        diff_w = diff @ self.conductivity                            # (D, 3)
        dist = torch.norm(diff, dim=1, keepdim=True) + 1e-6          # (D, 1)
        # clamp 保证物理合理 (近场 1.5 ~ 远场 4.5)，超出范围 forward 仍 OK
        p = self.decay_exponent.clamp(1.5, 4.5)
        return diff_w / (dist ** p)

    def forward(self, moments, x_in=None):
        """
        moments: (B, num_dipoles*3, L)   编码器输出的等效心电偶极矩
        x_in   : (B, num_input_leads, L) 测量到的输入导联 (I, II, V2)。
                 给定时启用 【改 3a】: 用真实 I/II 代数推 III/aVR/aVL/aVF,
                 把 moment 自由度全留给胸前导联。
        """
        B, _, L = moments.shape
        P = moments.view(B, self.num_dipoles, 3, L)

        ra = self.RA_base + self.limb_offset[0]
        la = self.LA_base + self.limb_offset[1]
        ll = self.LL_base + self.limb_offset[2]
        chest = self.chest_base + self.chest_offset

        V12 = torch.zeros(B, 12, L, device=moments.device, dtype=moments.dtype)

        if x_in is not None:
            # 【改 3a】肢体派生导联走严格代数等式 (Einthoven / Goldberger)。
            # 编码器不再被迫拟合 I/II → 6~9 个 moment 通道全部投到 V1~V6。
            I_in  = x_in[:, 0, :]
            II_in = x_in[:, 1, :]
            V12[:, 0, :] = I_in
            V12[:, 1, :] = II_in
            V12[:, 2, :] = II_in - I_in              # III
            V12[:, 3, :] = -(I_in + II_in) / 2.0     # aVR
            V12[:, 4, :] = I_in - II_in / 2.0        # aVL
            V12[:, 5, :] = II_in - I_in / 2.0        # aVF
        else:
            # 兼容路径：从物理生成的 I_sig / II_sig 推
            A_LA = self._transfer(la)
            A_RA = self._transfer(ra)
            A_LL = self._transfer(ll)
            I_sig  = torch.einsum('dc, bdcl -> bl', A_LA - A_RA, P) * self.lead_gain[0]
            II_sig = torch.einsum('dc, bdcl -> bl', A_LL - A_RA, P) * self.lead_gain[1]
            V12[:, 0, :] = I_sig
            V12[:, 1, :] = II_sig
            V12[:, 2, :] = II_sig - I_sig
            V12[:, 3, :] = -(I_sig + II_sig) / 2.0
            V12[:, 4, :] = I_sig - II_sig / 2.0
            V12[:, 5, :] = II_sig - I_sig / 2.0

        for i in range(6):
            A_c = self._transfer(chest[i])
            V12[:, 6 + i, :] = torch.einsum('dc, bdcl -> bl', A_c, P) * self.lead_gain[2 + i]

        return V12


class ResidualRefineHead(nn.Module):
    """
    【改 4】物理 + 残差精修头 (Hybrid)
    输入: 物理头输出 (B, 12, L) ⊕ 输入导联 (B, num_input_leads, L)  →  通道维 cat
    输出: 12 路残差 Δ，叠加到物理输出上

    关键: 最后一层 1x1 conv 的权重 / bias 初始化为 0，
          这样训练开始时 Δ ≡ 0，整体输出 == 纯物理预测，
          后续训练只学"物理预测的系统偏差"，先验有保护。
    """
    def __init__(self, num_input_leads=3, ch=64, kernel_size=7):
        super().__init__()
        in_ch = 12 + num_input_leads
        self.refine = nn.Sequential(
            nn.Conv1d(in_ch, ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            ResidualBlock(ch, kernel_size=kernel_size),
            nn.Conv1d(ch, 12, 1),
        )
        # zero init 最后一层，保证起跑等于物理头
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, phys_out, x_in):
        return self.refine(torch.cat([phys_out, x_in], dim=1))


class ECPINN_3D(nn.Module):
    # 【修改】默认 num_input_leads 改为 3
    def __init__(self, num_dipoles=3, num_input_leads=3, use_physics=True,
                 input_lead_indices=(0, 1, 7),
                 use_residual_refine=False,
                 refine_channels=64,
                 refine_kernel_size=7):
        """
        input_lead_indices:
            输入导联在 PTB 12 导联中的位置 (默认 (0, 1, 7) 对应 I, II, V2)。
            forward 末尾会把这些位置的输出 **直接替换为测量到的输入信号**。
        use_residual_refine:
            是否在物理头后追加 ResidualRefineHead (Hybrid)。
            仅在 use_physics=True 时有效；纯 CNN 路径保持"纯"以维持公平基线。
        """
        super().__init__()
        self.num_dipoles = num_dipoles
        self.num_input_leads = num_input_leads
        self.use_physics = use_physics

        if len(input_lead_indices) != num_input_leads:
            raise ValueError(
                f"input_lead_indices 长度 ({len(input_lead_indices)}) 必须等于 "
                f"num_input_leads ({num_input_leads})"
            )
        # 用 persistent=False 的 buffer：随模型迁移到 cuda/cpu，但不进 state_dict，
        # 这样新老权重的 state_dict 完全兼容
        self.register_buffer(
            "_input_lead_idx",
            torch.tensor(list(input_lead_indices), dtype=torch.long),
            persistent=False,
        )

        out_ch = (num_dipoles * 3) if use_physics else 12

        self.encoder = nn.Sequential(
            nn.Conv1d(num_input_leads, 64, 15, padding=7),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock(64, kernel_size=11),
            ChannelAttention(64),

            nn.Conv1d(64, 128, 7, padding=3),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            ResidualBlock(128, kernel_size=7),
            ChannelAttention(128),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            ResidualBlock(128, kernel_size=5),

            nn.Conv1d(128, out_ch, 1),
        )

        self.physics = Physics3DLayer(num_dipoles) if use_physics else None

        # 【改 4】残差精修头. 可以独立于物理使用 ——
        #   - use_physics=True  + use_residual_refine=True  : 物理 + Hybrid 残差
        #   - use_physics=False + use_residual_refine=True  : 纯 CNN + 残差 (ablation: 隔离 refine head 的容量贡献)
        #   - use_physics=True  + use_residual_refine=False : 纯物理
        #   - use_physics=False + use_residual_refine=False : 纯 CNN baseline
        self.use_residual_refine = bool(use_residual_refine)
        if self.use_residual_refine:
            self.refine_head = ResidualRefineHead(
                num_input_leads=num_input_leads,
                ch=refine_channels,
                kernel_size=refine_kernel_size,
            )
        else:
            self.refine_head = None

    def forward(self, x_in):
        z = self.encoder(x_in)
        if self.use_physics:
            # 物理头里启用 "用真实 I/II 代数推肢体派生导联" 路径 (改 3a)
            out = self.physics(z, x_in=x_in)
        else:
            out = z
        if self.refine_head is not None:
            # 【改 4】叠加残差: 起始 Δ=0 (zero-init), 随训练学"中间输出 → 真 12 路"的修正
            out = out + self.refine_head(out, x_in)

        # 【改 2】输入导联直通
        # 测量到的 I, II, V2 直接写回输出位置 (0, 1, 7)。
        # 效果：
        #   1) 输入导联损失恒为 0，编码器/物理头不再被迫去拟合 I/II/V2，
        #      物理头的 9 个 moment 自由度全部留给真正难重建的 9 路；
        #   2) 公平对比基线 —— 基线原本能走恒等捷径白拿 3 路 PCC，
        #      现在两种模型在这 3 个位置上的 PCC 都恒等于 1。
        B, _, L = out.shape
        idx_exp = self._input_lead_idx.view(1, -1, 1).expand(B, -1, L)
        out = out.scatter(1, idx_exp, x_in)
        return out


def infer_head_config_from_state_dict(state_dict):
    """
    扫描 state_dict 推断模型构造参数, 返回一个可以直接 **cfg 给 ECPINN_3D 用的 dict。
    支持: 纯 CNN / 纯物理 / 物理+残差精修 三种结构, 老权重也兼容。
    """
    encoder_keys = [k for k in state_dict.keys() if k.startswith("encoder.") and k.endswith(".weight")]
    if not encoder_keys:
        raise ValueError("无法在 state_dict 中找到 encoder 权重。")

    last_layer_key = sorted(encoder_keys, key=lambda x: int(x.split('.')[1]))[-1]
    w = state_dict.get(last_layer_key)
    if w.ndim != 3 or w.shape[1] != 128:
        raise ValueError(f"推断失败：最后一层权重 '{last_layer_key}' 的输入通道不是 128, 或非 1D 卷积格式")

    out_c = int(w.shape[0])
    has_physics = any(k.startswith("physics.") for k in state_dict)
    has_refine  = any(k.startswith("refine_head.") for k in state_dict)

    cfg = {
        "use_physics": False,
        "num_dipoles": 3,
        "use_residual_refine": has_refine,
        "refine_channels": 64,
        "refine_kernel_size": 7,
    }

    if has_physics:
        cfg["use_physics"] = True
        dp = state_dict.get("physics.dipole_pos")
        if dp is not None:
            cfg["num_dipoles"] = int(dp.shape[0])
        elif out_c % 3 == 0:
            cfg["num_dipoles"] = out_c // 3
        else:
            raise ValueError(f"权重含 physics 层但 encoder 输出通道 {out_c} 不是 3 的倍数")
    else:
        if out_c == 12:
            cfg["use_physics"] = False
        elif out_c % 3 == 0:
            cfg["use_physics"] = True
            cfg["num_dipoles"] = out_c // 3
        else:
            raise ValueError(f"encoder 输出通道数 {out_c} 既不是 12 也不是 3 的倍数")

    if has_refine:
        # refine_head.refine.0 是首个 Conv1d, weight shape = (ch, in_ch, kernel)
        w0 = state_dict.get("refine_head.refine.0.weight")
        if w0 is not None and w0.ndim == 3:
            cfg["refine_channels"] = int(w0.shape[0])
            cfg["refine_kernel_size"] = int(w0.shape[2])

    return cfg


def infer_num_dipoles_from_state_dict(state_dict):
    cfg = infer_head_config_from_state_dict(state_dict)
    if not cfg["use_physics"]:
        raise ValueError("该权重为无偶极子物理头模型。")
    return cfg["num_dipoles"]