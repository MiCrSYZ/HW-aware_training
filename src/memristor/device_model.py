"""
Memristor device model implementation.

This module provides the MemristorDeviceModel class that simulates
non-ideal memristor behavior including variability, read noise, drift,
stuck-at faults, and IR-drop effects.

Enhanced with:
- State-dependent write update model
- Pulse train writing
- ADC quantization
- Array size tiling
- Energy estimation
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List, Any


class MemristorDeviceModel:
    """
    Memristor device model with configurable non-idealities.
    
    This class maps software weights to conductance values and applies
    various non-idealities to simulate realistic memristor behavior.
    
    Args:
        G_min: Minimum conductance value (Siemens)
        G_max: Maximum conductance value (Siemens)
        weight_clip: Tuple (wmin, wmax) for weight clipping before mapping
        variability_sigma: Standard deviation of multiplicative variability (fractional)
        read_noise_sigma: Standard deviation of additive read noise (Siemens)
        drift_alpha: Drift coefficient (higher = more drift over time)
        stuck_ratio: Fraction of devices that are stuck-at (0.0 to 1.0)
        stuck_low_prob: Probability that stuck device is stuck at G_min (vs G_max)
        ir_drop_beta: IR-drop coefficient (0.0 to 1.0, higher = more voltage drop)
        mapping: Mapping strategy ('linear' or 'log')
        seed: Random seed for reproducibility (None for random)
    """
    
    def __init__(
        self,
        G_min: float = 1e-6,    #电导矩阵
        G_max: float = 1e-4,
        weight_clip: Tuple[float, float] = (-1.0, 1.0), #限制权重映射范围
        variability_sigma: float = 0.05,    #器件变异：每个单元的真实电导在目标值附近的偏移
        read_noise_sigma: float = 1e-7, #读噪声
        drift_alpha: float = 1e-4,  #电导漂移
        stuck_ratio: float = 0.0,   #固定值故障
        stuck_low_prob: float = 0.5,    #固定在低阻值的概率
        ir_drop_beta: float = 0.0,  #全局干扰缩放系数
        # linear: G=G_min+(G_max-G_min)*(W-W_min)/(W_max-W_min)
        # log: G=exp(a*W+b)
        mapping: str = 'linear',    # 'linear' or 'log'
        seed: Optional[int] = None,
        array_size: int = 128,  # 忆阻器阵列规模（tile大小）
        adc_bits: int = 8,  # ADC量化位数
        enable_update_model: bool = False,  # 启用状态依赖写入模型
        enable_adc: bool = False,  # 启用ADC量化
        enable_energy: bool = False,  # 启用能耗估计
        # 写入更新模型参数
        update_params: Optional[Dict[str, float]] = None,
        # 能耗系数
        energy_coefs: Optional[Dict[str, float]] = None,
        # 电导漂移时间设置方式
        drift_time_mode: str = 'accumulate',  # 'fixed' 或 'accumulate'
        drift_time_fixed: int = 0,  # 固定值模式下的t值
        # 新的IR-drop模型参数（基于论文方程16-18）
        ir_drop_mode: str = 'none',  # 'none' | 'simple' | 'paper'
        ir_drop_gamma: float = 0.35,  # 导线电阻缩放因子
        ir_drop_scaling: float = 1.0,  # IR-drop校正项的最终缩放因子
    ):
        self.G_min = G_min
        self.G_max = G_max
        self.wmin, self.wmax = weight_clip
        self.variability_sigma = variability_sigma
        self.read_noise_sigma = read_noise_sigma
        self.drift_alpha = drift_alpha
        self.stuck_ratio = stuck_ratio
        self.stuck_low_prob = stuck_low_prob
        self.ir_drop_beta = ir_drop_beta
        self.mapping = mapping
        
        # 新增参数
        self.array_size = array_size
        self.adc_bits = adc_bits
        self.enable_update_model = enable_update_model
        self.enable_adc = enable_adc
        self.enable_energy = enable_energy
        
        # 写入更新模型参数（默认值）
        if update_params is None:
            update_params = {
                'A_plus': 1e-5,  # Potentiation幅度系数
                'A_minus': 1e-5,  # Depression幅度系数
                'p_plus': 1.0,  # Potentiation非线性指数
                'p_minus': 1.0,  # Depression非线性指数
                'gamma': 1.0,  # 电压-时间耦合系数
                'write_noise_ratio': 0.05,  # 写入噪声比例（相对于ΔG）
            }
        self.update_params = update_params
        
        # 能耗系数（默认值）
        if energy_coefs is None:
            energy_coefs = {
                'alpha': 1.0,  # 写入能耗系数 (E_write = alpha * V^2 * t)
                'beta': 1.0,  # 读出能耗系数 (E_read = beta * 2^bits)
            }
        self.energy_coefs = energy_coefs
        
        # 能耗统计字典
        self.energy_stats: Dict[str, float] = {
            'write': 0.0,
            'read': 0.0,
        }
        
        # 电导漂移时间设置方式
        # 'fixed': 使用固定值 drift_time_fixed
        # 'accumulate': 模拟真实器件老化，t随推理次数累加
        self.drift_time_mode = drift_time_mode
        self.drift_time_fixed = drift_time_fixed
        self.inference_count: int = 0  # 累加模式下的推理次数计数器
        
        # 新的IR-drop模型参数
        self.ir_drop_mode = ir_drop_mode
        self.ir_drop_gamma = ir_drop_gamma
        self.ir_drop_scaling = ir_drop_scaling
        
        if seed is not None:
            torch.manual_seed(seed)
    
    def map_weights_to_conductance(
        self, 
        W: torch.Tensor,
        learned_scale: Optional[float] = None,
        learned_offset: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Map software weight matrix to conductance values.
        
        Args:
            W: Weight tensor of any shape
            learned_scale: Optional learned scale parameter for adaptive mapping
            learned_offset: Optional learned offset parameter for adaptive mapping
            
        Returns:
            G: Conductance tensor with same shape as W
        """
        W_clamped = torch.clamp(W, self.wmin, self.wmax)
        
        # Apply learned mapping parameters if provided
        if learned_scale is not None:
            W_clamped = W_clamped * learned_scale
        if learned_offset is not None:
            W_clamped = W_clamped + learned_offset
        # Re-clamp after learned mapping
        W_clamped = torch.clamp(W_clamped, self.wmin, self.wmax)
        
        if self.mapping == 'linear':
            # Linear mapping: [wmin, wmax] -> [G_min, G_max]
            scale = (W_clamped - self.wmin) / (self.wmax - self.wmin + 1e-12)
            G = self.G_min + scale * (self.G_max - self.G_min)
        elif self.mapping == 'log':
            # Logarithmic mapping: G = exp(a*W + b)
            a = (np.log(self.G_max) - np.log(self.G_min)) / (self.wmax - self.wmin + 1e-12)
            b = np.log(self.G_min) - a * self.wmin
            G = torch.exp(a * W_clamped + b)
        else:
            raise ValueError(f"Unknown mapping: {self.mapping}. Use 'linear' or 'log'.")
        
        return G

    def map_weights_to_conductance_diff_adaptive(self, W: torch.Tensor, eps: float = 1e-12):
        """
        Differential pair mapping with per-layer adaptive normalization.
        
        This function implements the corrected mapping that fixes the numerical scale issue:
        1. Clamps weights to [wmin, wmax]
        2. Computes per-layer max_abs = max(|W|) for normalization
        3. Normalizes positive part: W_pos = max(W, 0) / max_abs
        4. Normalizes negative part: W_neg = max(-W, 0) / max_abs
        5. Maps both to full conductance range: G = G_min + (W_norm * (G_max - G_min))
        
        This ensures that the full weight range is mapped to the full conductance range,
        avoiding the issue where small conductance values (~1e-6) cause output magnitudes
        to be many orders of magnitude smaller than expected.
        
        Args:
            W: Weight tensor of any shape
            eps: Small epsilon to avoid division by zero
            
        Returns:
            G_pos: Positive conductance tensor with same shape as W
            G_neg: Negative conductance tensor with same shape as W
            max_abs: Per-layer maximum absolute value (scalar tensor) for scale recovery
        """
        # Clamp weights to configured range
        W_clamped = torch.clamp(W, self.wmin, self.wmax)

        # Compute per-layer scalar: use absolute max over whole weight matrix
        max_abs = W_clamped.abs().max().clamp_min(eps)  # avoid zero division
        
        # Positive / negative parts normalized by max_abs
        # This ensures the full weight range [0, max_abs] maps to [0, 1]
        W_pos_norm = torch.clamp(W_clamped, min=0.0) / max_abs
        W_neg_norm = torch.clamp(-W_clamped, min=0.0) / max_abs

        # Map normalized [0, 1] to full conductance range [G_min, G_max]
        G_range = (self.G_max - self.G_min)
        G_pos = self.G_min + W_pos_norm * G_range
        G_neg = self.G_min + W_neg_norm * G_range

        # Return conductances and max_abs for scale recovery in forward pass
        return G_pos, G_neg, max_abs

    def apply_nonidealities(
        self,
        G: torch.Tensor,
        t: int = 0,
        seed: Optional[int] = None,
        col_load: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply non-idealities to conductance values.
        
        This is the key function where device non-idealities are injected.
        Applied effects:
        1. Variability: Multiplicative Gaussian noise
        2. Read noise: Additive Gaussian noise
        3. Drift: Time-dependent degradation
        4. Stuck-at faults: Devices stuck at G_min or G_max
        5. IR-drop: Column-dependent voltage drop
        
        Args:
            G: Conductance tensor
            t: Time/cycle index for drift calculation
            seed: Random seed for this operation (None for random)
            col_load: Optional column load tensor for IR-drop calculation
            
        Returns:
            G_noisy: Conductance tensor with non-idealities applied
        """
        device = G.device
        dtype = G.dtype
        
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        # G'=G*(1+epsilon_v)+eta
        # epsilon_v~N(0, sigma_v), eta~N(0, sigma_read)
        # 1. 各器件间差异(multiplicative)
        if self.variability_sigma > 0:
            if generator is not None:
                eps = torch.randn(G.size(), generator=generator, device=device, dtype=dtype, requires_grad=False) * self.variability_sigma
            else:
                eps = torch.randn_like(G) * self.variability_sigma
            # Random noise should not have gradients, but multiplication with G maintains gradient flow
            G = G * (1.0 + eps)
        
        # 2. 读噪声(additive)
        if self.read_noise_sigma > 0:
            if generator is not None:
                noise = torch.randn(G.size(), generator=generator, device=device, dtype=dtype, requires_grad=False) * self.read_noise_sigma
            else:
                noise = torch.randn_like(G) * self.read_noise_sigma
            # Random noise should not have gradients, but addition with G maintains gradient flow
            G = G + noise
        
        # 3. 随时间变化的电导漂移
        # G_t=G'*(1-alpha*log(1+t))
        # 根据配置选择时间t的设置方式
        if self.drift_alpha > 0:
            if self.drift_time_mode == 'fixed':
                # 固定值模式：使用配置的固定值
                drift_t = self.drift_time_fixed
            elif self.drift_time_mode == 'accumulate':
                # 累加模式：使用推理次数计数器（模拟真实器件老化）
                drift_t = self.inference_count
            else:
                # 默认：使用传入的t值（向后兼容）
                drift_t = t
            
            if drift_t > 0:
                drift_factor = 1.0 - self.drift_alpha * np.log(1 + drift_t)
                drift_factor = max(drift_factor, 0.0)
                G = G * drift_factor
        
        # 4. 固定值故障：单元固定在高/低阻值
        if self.stuck_ratio > 0:
            if generator is not None:
                mask = torch.rand(G.size(), generator=generator, device=device, dtype=dtype, requires_grad=False) < self.stuck_ratio
                stuck_rand = torch.rand(G.size(), generator=generator, device=device, dtype=dtype, requires_grad=False)
            else:
                mask = torch.rand_like(G) < self.stuck_ratio
                stuck_rand = torch.rand_like(G)
            
            if mask.any():
                stuck_low = (stuck_rand < self.stuck_low_prob) & mask
                stuck_high = mask & (~stuck_low)
                # torch.where maintains gradient flow: gradient only flows through non-stuck elements
                G = torch.where(stuck_low, torch.full_like(G, self.G_min), G)
                G = torch.where(stuck_high, torch.full_like(G, self.G_max), G)
        
        # 5. IR-drop: 交叉阵列中因导线电阻导致的电压降
        if self.ir_drop_beta > 0:
            if col_load is not None:
                # Normalize col_load to [0, 1]
                col = torch.tensor(col_load, device=device, dtype=dtype)
                col_min = col.min()
                col_max = col.max()
                if col_max > col_min:
                    col_norm = (col - col_min) / (col_max - col_min + 1e-12)
                else:
                    col_norm = torch.zeros_like(col)
                
                # Expand to match G dimensions
                scale = 1.0 - self.ir_drop_beta * col_norm
                while scale.dim() < G.dim():
                    scale = scale.unsqueeze(0)
                # Broadcast scale to match G shape
                for _ in range(G.dim() - scale.dim()):
                    scale = scale.unsqueeze(-1)
                G = G * scale
            else:
                # Simplified: uniform scaling
                G = G * (1.0 - self.ir_drop_beta)
        
        # Ensure conductance stays in valid range
        G = torch.clamp(G, self.G_min, self.G_max)
        
        return G
    
    def apply_ir_drop_paper(
        self,
        y_tilde: torch.Tensor,
        W_tilde: torch.Tensor,
        x_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于论文方程(16)-(18)的IR-drop校正实现。
        
        实现方程：
        1) a_i = γ * n * sum_j |ŵ_ij| * |x̃_j|
        2) c_i = 0.05*a_i^3 – 0.2*a_i^2 + 0.5*a_i
        3) Δỹ_i = -c_i * sum_j ŵ_ij * x̃_j * ( j^2 / n^2 )
        
        Args:
            y_tilde: 归一化输出 [batch, out_features]
            W_tilde: 归一化权重 [out_features, in_features]
            x_tilde: 归一化输入 [batch, in_features]
            
        Returns:
            y_tilde_with_ir: 应用IR-drop校正后的输出 [batch, out_features]
        """
        batch_size, out_features = y_tilde.shape
        in_features = x_tilde.shape[1]
        n = in_features  # 输入列数
        
        device = y_tilde.device
        dtype = y_tilde.dtype
        
        # 计算位置向量 j^2 / n^2，其中 j 从 1 到 n
        # j = [1, 2, 3, ..., n]
        j_indices = torch.arange(1, n + 1, device=device, dtype=dtype)  # [n]
        position_factor = (j_indices ** 2) / (n ** 2)  # [n]
        
        # 计算 |ŵ_ij| * |x̃_j| 对每个输出神经元i和批次
        # W_tilde: [out_features, in_features]
        # x_tilde: [batch, in_features]
        abs_W_tilde = torch.abs(W_tilde)  # [out_features, in_features]
        abs_x_tilde = torch.abs(x_tilde)  # [batch, in_features]
        
        # 计算 sum_j |ŵ_ij| * |x̃_j| 对每个输出神经元i和批次
        # 使用矩阵乘法: [batch, out_features] = [batch, in_features] @ [in_features, out_features]
        activity_sum = torch.matmul(abs_x_tilde, abs_W_tilde.T)  # [batch, out_features]
        
        # 计算活动项 a_i = γ * n * sum_j |ŵ_ij| * |x̃_j|
        a_i = self.ir_drop_gamma * n * activity_sum  # [batch, out_features]
        
        # 计算非线性多项式系数 c_i = 0.05*a_i^3 – 0.2*a_i^2 + 0.5*a_i
        c_i = (0.05 * (a_i ** 3) - 
               0.2 * (a_i ** 2) + 
               0.5 * a_i)  # [batch, out_features]
        
        # 计算 sum_j ŵ_ij * x̃_j * ( j^2 / n^2 )
        # 优化：使用矩阵乘法而不是扩展维度
        # 首先计算 x_tilde * position_factor: [batch, in_features]
        x_tilde_weighted = x_tilde * position_factor.unsqueeze(0)  # [batch, in_features]
        
        # 然后计算 W_tilde @ (x_tilde_weighted).T: [out_features, batch]
        # 转置得到: [batch, out_features]
        weighted_sum = torch.matmul(x_tilde_weighted, W_tilde.T)  # [batch, out_features]
        
        # 计算IR-drop校正项 Δỹ_i = -c_i * sum_j ŵ_ij * x̃_j * ( j^2 / n^2 )
        delta_y_tilde = -c_i * weighted_sum  # [batch, out_features]
        
        # 应用最终缩放因子
        delta_y_tilde = delta_y_tilde * self.ir_drop_scaling
        
        # 返回校正后的输出 ỹ + Δỹ
        y_tilde_with_ir = y_tilde + delta_y_tilde
        
        return y_tilde_with_ir
    
    def write_update(
        self,
        G: torch.Tensor,
        pulse_V: torch.Tensor,
        pulse_t: torch.Tensor,
        direction: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        状态依赖的写入更新模型。
        
        Args:
            G: 当前电导值张量
            pulse_V: 脉冲电压张量（与G同形状）
            pulse_t: 脉冲宽度张量（与G同形状，单位：秒）
            direction: 方向张量（与G同形状，>0表示potentiation，<=0表示depression）
            seed: 随机种子
            
        Returns:
            G_new: 更新后的电导值张量
        """
        if not self.enable_update_model:
            # 如果未启用更新模型，直接返回原值
            return G
        
        device = G.device
        dtype = G.dtype
        
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None
        
        # 提取参数并转换为与输入张量兼容的类型
        device = G.device
        dtype = G.dtype
        
        A_plus = float(self.update_params['A_plus'])   #A_plus和A_minus不同以表示非对称性
        A_minus = float(self.update_params['A_minus'])
        p_plus = float(self.update_params['p_plus'])   #p_plus和p_minus控制饱和速度(>1时饱和得更快)
        p_minus = float(self.update_params['p_minus'])
        gamma = float(self.update_params['gamma'])
        noise_ratio = float(self.update_params['write_noise_ratio'])   #写噪声
        
        # 脉冲驱动函数: f(V,t)=1 - exp(-γ*V*t)
        exp_term = 1.0 - torch.exp(-gamma * pulse_V * pulse_t)
        
        # 增强: ΔG_pot = A⁺*(1 − G/G_max)^(p⁺)*f(V,t)
        pot_mask = direction > 0
        G_norm_pot = 1.0 - (G / (self.G_max + 1e-12))
        G_norm_pot = torch.clamp(G_norm_pot, min=0.0, max=1.0)  # 确保在[0,1]范围内
        # 使用 torch.pow 而不是 ** 运算符，确保类型兼容
        delta_G_pot = A_plus * torch.pow(G_norm_pot, p_plus) * exp_term
        
        # 抑制: ΔG_dep = −A⁻ *((G/G_min − 1))^(p⁻)*f(V,t)
        dep_mask = direction <= 0
        G_norm_dep = (G / (self.G_min + 1e-12)) - 1.0
        G_norm_dep = torch.clamp(G_norm_dep, min=0.0, max=1.0)  # 确保在[0,1]范围内
        # 使用 torch.pow 而不是 ** 运算符，确保类型兼容
        delta_G_dep = -A_minus * torch.pow(G_norm_dep, p_minus) * exp_term
        
        # 组合更新
        delta_G = torch.where(pot_mask, delta_G_pot, delta_G_dep)
        
        # 添加写入噪声(高斯噪声)：noise = 0.05*|ΔG|*N(0,1)
        if noise_ratio > 0:
            if generator is not None:
                noise = torch.randn(G.size(), generator=generator, device=device, dtype=dtype) * noise_ratio * torch.abs(delta_G)
            else:
                noise = torch.randn_like(G) * noise_ratio * torch.abs(delta_G)
            delta_G = delta_G + noise
        
        # 更新电导值
        G_new = G + delta_G
        
        # 限制在有效范围内
        G_new = torch.clamp(G_new, self.G_min, self.G_max)
        
        # 累积写入能耗
        if self.enable_energy:
            # E_write = alpha * V^2 * t
            alpha = self.energy_coefs['alpha']
            energy_write = alpha * (pulse_V ** 2) * pulse_t
            self.energy_stats['write'] += float(energy_write.sum().detach().cpu().item())
        
        return G_new
    
    def write_pulse_train(
        self,
        G: torch.Tensor,
        V_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
        direction_list: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        脉冲序列写入模型。
        
        逐个应用脉冲，累积更新电导值。
        
        Args:
            G: 初始电导值张量
            V_list: 脉冲电压列表，每个元素是与G同形状的张量
            t_list: 脉冲宽度列表，每个元素是与G同形状的张量
            direction_list: 方向列表（可选），如果为None，则根据V的符号自动判断
            seed: 随机种子（每个脉冲会递增）
            
        Returns:
            G_final: 最终电导值张量
        """
        if not self.enable_update_model:
            return G
        
        G_current = G.clone()
        current_seed = seed
        
        for i, (V, t) in enumerate(zip(V_list, t_list)):
            # 如果没有提供direction_list，根据V的符号自动判断
            if direction_list is None:
                direction = torch.sign(V)  # >0为potentiation，<0为depression
            else:
                direction = direction_list[i]
            
            # 应用单个脉冲更新
            if current_seed is not None:
                pulse_seed = current_seed + i
            else:
                pulse_seed = None
            
            G_current = self.write_update(G_current, V, t, direction, seed=pulse_seed)
        
        return G_current
    
    def adc_quant(
        self,
        x: torch.Tensor,
        bits: Optional[int] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        """
        ADC量化函数。
        
        使用min-max scaling进行量化：
        1. 计算min和max
        2. 缩放到[0, 2^bits - 1]
        3. 四舍五入
        4. 缩放回原范围
        
        Args:
            x: 输入张量
            bits: ADC位数（如果为None，使用self.adc_bits）
            add_noise: 是否添加轻微的量化噪声
            
        Returns:
            x_quant: 量化后的张量
        """
        if not self.enable_adc:
            return x
        
        if bits is None:
            bits = self.adc_bits
        
        device = x.device
        dtype = x.dtype
        
        # 计算min和max（按最后一个维度，通常是列方向）
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        
        # 避免除零
        x_range = x_max - x_min
        x_range = torch.clamp(x_range, min=1e-12)
        
        # 缩放到[0, 2^bits - 1]
        scale = (2.0 ** bits - 1.0) / x_range
        x_scaled = (x - x_min) * scale
        
        # 四舍五入
        x_quantized = torch.round(x_scaled)
        
        # 缩放回原范围
        x_quant = x_quantized / scale + x_min
        
        # 可选：添加轻微的量化噪声
        if add_noise:
            noise_scale = x_range / (2.0 ** bits)
            noise = torch.randn_like(x) * noise_scale * 0.1  # 10%的量化步长作为噪声
            x_quant = x_quant + noise
        
        return x_quant
    
    def matmul_with_tiling(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        adc_bits: Optional[int] = None,
    ) -> torch.Tensor:
        """
        带tiling和ADC量化的矩阵乘法。
        
        将大矩阵按array_size分块，每个tile的输出经过ADC量化，然后累加。
        
        Args:
            x: 输入张量 [batch, in_dim]
            W: 权重矩阵 [out_dim, in_dim]
            adc_bits: ADC位数（如果为None，使用self.adc_bits）
            
        Returns:
            y: 输出张量 [batch, out_dim]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)

        assert W.shape[1] == x.shape[1], \
            f"[Error] W.shape={W.shape}, x.shape={x.shape}, in_dim mismatch"

        if adc_bits is None:
            adc_bits = self.adc_bits
        
        batch_size = x.size(0)
        out_dim, in_dim = W.size()
        array_size = self.array_size
        
        # 初始化输出
        y = torch.zeros(batch_size, out_dim, device=x.device, dtype=x.dtype)
        
        # 按tile进行矩阵乘法
        num_tiles = 0
        for r0 in range(0, out_dim, array_size):
            r1 = min(r0 + array_size, out_dim)
            for c0 in range(0, in_dim, array_size):
                c1 = min(c0 + array_size, in_dim)
                
                # 提取tile
                W_tile = W[r0:r1, c0:c1]  # [tile_out, tile_in]
                x_tile = x[:, c0:c1]  # [batch, tile_in]

                assert W_tile.shape[1] == x_tile.shape[1], \
                    f"Tile mismatch: W_tile={W_tile.shape}, x_tile={x_tile.shape}"
                
                # 矩阵乘法
                y_partial = torch.matmul(x_tile, W_tile.T)  # [batch, tile_out]
                
                # ADC量化
                if self.enable_adc:
                    y_partial = self.adc_quant(y_partial, bits=adc_bits)
                
                # 累加到输出
                y[:, r0:r1] = y[:, r0:r1] + y_partial
                
                num_tiles += 1
        
        # 累积读出能耗
        if self.enable_energy:
            # E_read = num_tiles * beta * 2^bits
            beta = self.energy_coefs['beta']
            energy_read = num_tiles * beta * (2.0 ** adc_bits)
            self.energy_stats['read'] += energy_read

        if len(orig_shape) == 3:
            y = y.reshape(B, L, -1)

        return y
    
    def reset_energy_stats(self) -> None:
        """重置能耗统计。"""
        self.energy_stats = {
            'write': 0.0,
            'read': 0.0,
        }
    
    def reset_inference_count(self) -> None:
        """重置推理次数计数器（用于累加模式的电导漂移）。"""
        self.inference_count = 0
    
    def increment_inference_count(self) -> None:
        """增加推理次数计数器（用于累加模式的电导漂移）。
        
        应该在每次前向传播开始时调用一次。
        """
        if self.drift_time_mode == 'accumulate':
            self.inference_count += 1
    
    def get_energy_stats(self) -> Dict[str, float]:
        """获取当前能耗统计。"""
        return self.energy_stats.copy()
    
    def save_state(self, path: str) -> None:
        """
        Save device model state to file.
        
        Args:
            path: Path to save state
        """
        state = {
            'G_min': self.G_min,
            'G_max': self.G_max,
            'wmin': self.wmin,
            'wmax': self.wmax,
            'variability_sigma': self.variability_sigma,
            'read_noise_sigma': self.read_noise_sigma,
            'drift_alpha': self.drift_alpha,
            'stuck_ratio': self.stuck_ratio,
            'stuck_low_prob': self.stuck_low_prob,
            'ir_drop_beta': self.ir_drop_beta,
            'mapping': self.mapping,
            # 新增参数
            'array_size': self.array_size,
            'adc_bits': self.adc_bits,
            'enable_update_model': self.enable_update_model,
            'enable_adc': self.enable_adc,
            'enable_energy': self.enable_energy,
            'update_params': self.update_params,
            'energy_coefs': self.energy_coefs,
            'drift_time_mode': self.drift_time_mode,
            'drift_time_fixed': self.drift_time_fixed,
            # 新的IR-drop参数
            'ir_drop_mode': self.ir_drop_mode,
            'ir_drop_gamma': self.ir_drop_gamma,
            'ir_drop_scaling': self.ir_drop_scaling,
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """
        Load device model state from file.
        
        Args:
            path: Path to load state from
        """
        state = torch.load(path)
        self.G_min = state['G_min']
        self.G_max = state['G_max']
        self.wmin = state['wmin']
        self.wmax = state['wmax']
        self.variability_sigma = state['variability_sigma']
        self.read_noise_sigma = state['read_noise_sigma']
        self.drift_alpha = state['drift_alpha']
        self.stuck_ratio = state['stuck_ratio']
        self.stuck_low_prob = state['stuck_low_prob']
        self.ir_drop_beta = state['ir_drop_beta']
        self.mapping = state['mapping']
        # 新增参数（向后兼容）
        self.array_size = state.get('array_size', 128)
        self.adc_bits = state.get('adc_bits', 6)
        self.enable_update_model = state.get('enable_update_model', False)
        self.enable_adc = state.get('enable_adc', False)
        self.enable_energy = state.get('enable_energy', False)
        self.update_params = state.get('update_params', {
            'A_plus': 1e-5, 'A_minus': 1e-5, 'p_plus': 1.0, 'p_minus': 1.0,
            'gamma': 1.0, 'write_noise_ratio': 0.05
        })
        self.energy_coefs = state.get('energy_coefs', {'alpha': 1.0, 'beta': 1.0})
        # 重置能耗统计
        self.energy_stats = {'write': 0.0, 'read': 0.0}
        # 电导漂移时间设置（向后兼容）
        self.drift_time_mode = state.get('drift_time_mode', 'accumulate')
        self.drift_time_fixed = state.get('drift_time_fixed', 0)
        self.inference_count = 0  # 重置推理次数计数器
        # 新的IR-drop参数（向后兼容）
        self.ir_drop_mode = state.get('ir_drop_mode', 'none')
        self.ir_drop_gamma = state.get('ir_drop_gamma', 0.35)
        self.ir_drop_scaling = state.get('ir_drop_scaling', 1.0)


