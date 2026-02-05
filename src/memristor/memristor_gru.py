"""
Memristor-aware GRU layer wrapper.

This module provides a GRU layer wrapper that applies memristor device
non-idealities to GRU weights and outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple

try:
    from .device_model import MemristorDeviceModel
    from .memristor_wrappers import hardware_linear_forward_adaptive
except ImportError:
    from src.memristor.device_model import MemristorDeviceModel
    from src.memristor.memristor_wrappers import hardware_linear_forward_adaptive


class MemristorGRU(nn.Module):
    """
    GRU layer with memristor device non-idealities.
    
    Noise injection strategy:
    - Weight-domain noise (variability, cond1): Applied to weight_ih and weight_hh
    - Output-domain noise (cond2, cond3): Applied to candidate gate output
    """
    
    def __init__(
        self,
        gru: nn.GRU,
        device_model: MemristorDeviceModel,
        enable_weight_noise: bool = True,
        enable_output_noise: bool = True,
    ):
        """
        Initialize memristor-aware GRU layer.
        
        Args:
            gru: Standard nn.GRU layer to wrap
            device_model: MemristorDeviceModel instance
            enable_weight_noise: Whether to inject weight-domain noise (variability, cond1)
            enable_output_noise: Whether to inject output-domain noise (cond2, cond3)
        """
        super().__init__()
        self.device_model = device_model
        self.enable_weight_noise = enable_weight_noise
        self.enable_output_noise = enable_output_noise
        
        # Copy GRU parameters
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        self.num_layers = gru.num_layers
        self.bias = gru.bias
        self.batch_first = gru.batch_first
        self.dropout = gru.dropout
        self.bidirectional = gru.bidirectional
        self.proj_size = gru.proj_size if hasattr(gru, 'proj_size') else 0
        
        # Copy weights
        # GRU has weight_ih_l[k] and weight_hh_l[k] for each layer k
        self._all_weights = []
        for layer in range(self.num_layers):
            for direction in range(2 if self.bidirectional else 1):
                layer_input_size = self.input_size if layer == 0 else (
                    self.hidden_size * (2 if self.bidirectional else 1)
                )
                
                suffix = '_reverse' if direction == 1 else ''
                weight_ih_name = f'weight_ih_l{layer}{suffix}'
                weight_hh_name = f'weight_hh_l{layer}{suffix}'
                
                # Copy weight_ih
                weight_ih = getattr(gru, weight_ih_name)
                self.register_parameter(weight_ih_name, nn.Parameter(weight_ih.data.clone()))
                
                # Copy weight_hh
                weight_hh = getattr(gru, weight_hh_name)
                self.register_parameter(weight_hh_name, nn.Parameter(weight_hh.data.clone()))
                
                # Copy biases if they exist
                if self.bias:
                    bias_ih_name = f'bias_ih_l{layer}{suffix}'
                    bias_hh_name = f'bias_hh_l{layer}{suffix}'
                    
                    bias_ih = getattr(gru, bias_ih_name)
                    bias_hh = getattr(gru, bias_hh_name)
                    
                    self.register_parameter(bias_ih_name, nn.Parameter(bias_ih.data.clone()))
                    self.register_parameter(bias_hh_name, nn.Parameter(bias_hh.data.clone()))
                
                self._all_weights.append([weight_ih_name, weight_hh_name])
    
    def _apply_weight_noise(
        self,
        weight: torch.Tensor,
        t: int = 0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply weight-domain noise (variability, cond1) to a weight tensor.
        
        This uses the same mechanism as MemristorLinear for weight-domain noise.
        """
        if not self.enable_weight_noise:
            return weight
        
        # Use hardware_linear_forward_adaptive logic for weight-domain noise
        # We only need the weight transformation part, not the forward pass
        Gp, Gn, max_abs = self.device_model.map_weights_to_conductance_diff_adaptive(weight)
        
        # Apply non-idealities based on synthetic_noise_type
        synthetic_noise_type = getattr(self.device_model, 'synthetic_noise_type', 'none')
        
        if synthetic_noise_type == 'full_variability':
            apply_nonidealities = True
        elif synthetic_noise_type == 'none':
            apply_nonidealities = self.enable_weight_noise
        else:
            # cond1/cond2/cond3: only cond1 applies weight-domain noise
            apply_nonidealities = (synthetic_noise_type == 'cond1_variance_bounded')
        
        if apply_nonidealities:
            Gp_noisy = self.device_model.apply_nonidealities(Gp, t=t, seed=seed)
            Gn_seed = seed if seed is None else seed + 1
            Gn_noisy = self.device_model.apply_nonidealities(Gn, t=t, seed=Gn_seed)
        else:
            Gp_noisy = Gp
            Gn_noisy = Gn
        
        # Convert back to weight scale
        G_range = self.device_model.G_max - self.device_model.G_min
        scale = max_abs / (G_range + 1e-12)
        scale = torch.clamp(scale, min=1e-3, max=1e6)
        W_eff = (Gp_noisy - Gn_noisy) * scale
        
        # Apply cond1 noise if enabled
        if synthetic_noise_type == 'cond1_variance_bounded':
            nu = self.device_model.cond1_nu
            if seed is not None:
                generator = torch.Generator(device=W_eff.device)
                generator.manual_seed(seed)
                Z = torch.randn(W_eff.shape, generator=generator, device=W_eff.device, dtype=W_eff.dtype, requires_grad=False)
                gamma_dist = torch.distributions.Gamma(concentration=nu / 2.0, rate=0.5)
                chi2 = gamma_dist.sample(W_eff.shape).to(W_eff.device)
                chi2 = torch.clamp(chi2, min=1e-8)
                t_noise = Z / torch.sqrt(chi2 / nu)
            else:
                t_dist = torch.distributions.StudentT(df=nu)
                t_noise = t_dist.sample(W_eff.shape).to(W_eff.device)
            
            alpha = self.device_model.cond1_alpha
            W_eff = W_eff * (1.0 + alpha * t_noise)
        
        return W_eff
    
    def _apply_output_noise(
        self,
        output: torch.Tensor,
        t: int = 0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply output-domain noise (cond2, cond3) to GRU gate outputs.
        
        This function applies noise to reset gate, update gate, and candidate gate outputs.
        
        Args:
            output: Gate output tensor (before activation function)
            t: Time/cycle index
            seed: Random seed
            
        Returns:
            Noisy output tensor
        """
        if not self.enable_output_noise:
            return output
        
        synthetic_noise_type = getattr(self.device_model, 'synthetic_noise_type', 'none')
        
        # Apply cond2 noise (gradient-unbiased)
        if synthetic_noise_type == 'cond2_gradient_unbiased':
            out_features = output.shape[-1]
            v_key = (out_features,)
            
            if v_key not in self.device_model._cond2_v_vectors:
                if self.device_model.seed is not None:
                    generator = torch.Generator(device=output.device)
                    generator.manual_seed(self.device_model.seed + hash(v_key) % 1000000)
                    v = torch.randn(out_features, generator=generator, device=output.device, dtype=output.dtype, requires_grad=False)
                else:
                    v = torch.randn(out_features, device=output.device, dtype=output.dtype, requires_grad=False)
                v = v / (torch.norm(v) + 1e-8)
                self.device_model._cond2_v_vectors[v_key] = v
            else:
                v = self.device_model._cond2_v_vectors[v_key].to(output.device)
            
            # Compute s(z) = 1 + α * tanh(v^T z)
            if output.dim() == 2:
                vTz = torch.sum(output * v.unsqueeze(0), dim=-1, keepdim=True)
            else:
                vTz = torch.sum(output * v.view(1, -1), dim=-1, keepdim=True)
            
            alpha = self.device_model.cond2_alpha
            s_z = 1.0 + alpha * torch.tanh(vTz)
            s_z_detached = s_z.detach()
            output = output * s_z_detached + (output * s_z - output * s_z_detached).detach()
        
        # Apply cond3 noise (ADC quantization)
        if synthetic_noise_type == 'cond3_adc_direct':
            if hasattr(self.device_model, 'enable_adc') and self.device_model.enable_adc:
                if self.training:
                    # Use direct mode for cond3
                    output_quantized = self.device_model.adc_quant(output, bits=self.device_model.adc_bits)
                    output = output_quantized.detach()
                else:
                    output = self.device_model.adc_quant(output, bits=self.device_model.adc_bits)
        
        return output
    
    def _gru_cell_forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
        t: int = 0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Single GRU cell forward pass with noise injection.
        
        Args:
            x_t: Input at time t [batch, input_size]
            h: Hidden state [batch, hidden_size]
            weight_ih: Input-to-hidden weights [3*hidden_size, input_size]
            weight_hh: Hidden-to-hidden weights [3*hidden_size, hidden_size]
            bias_ih: Input-to-hidden bias [3*hidden_size] or None
            bias_hh: Hidden-to-hidden bias [3*hidden_size] or None
            t: Time/cycle index
            seed: Random seed
            
        Returns:
            New hidden state [batch, hidden_size]
        """
        # Split weights into gates: reset (r), update (z), candidate (n)
        W_ir, W_iz, W_in = weight_ih.chunk(3, dim=0)
        W_hr, W_hz, W_hn = weight_hh.chunk(3, dim=0)
        
        if bias_ih is not None:
            b_ir, b_iz, b_in = bias_ih.chunk(3)
        else:
            b_ir = b_iz = b_in = None
        
        if bias_hh is not None:
            b_hr, b_hz, b_hn = bias_hh.chunk(3)
        else:
            b_hr = b_hz = b_hn = None
        
        # Check if cond3 is enabled
        synthetic_noise_type = getattr(self.device_model, 'synthetic_noise_type', 'none')
        is_cond3 = (self.enable_output_noise and synthetic_noise_type == 'cond3_adc_direct')
        
        # Debug: Print cond3 status (only once per training session)
        if not hasattr(self, '_cond3_debug_logged'):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[MemristorGRU] enable_output_noise={self.enable_output_noise}, synthetic_noise_type={synthetic_noise_type}, is_cond3={is_cond3}")
            if is_cond3:
                logger.warning(f"[MemristorGRU] cond3 mode detected! Will detach x_t and h to block GRU weight gradients")
            self._cond3_debug_logged = True
        
        # IMPORTANT: For cond3 (ADC quantization), we need to block gradients to GRU weights
        # The ADC quantization in _apply_output_noise detaches gate outputs, but gradients can still
        # flow through the F.linear operations BEFORE the detach. To properly block GRU weight gradients,
        # we detach x_t and h BEFORE using them in linear operations.
        # 
        # Note: This will also block gradients to embedding layer (since x_t is detached),
        # but cond3 mode is designed to simulate non-differentiable ADC quantization, so this is expected.
        h_for_gates = h.detach() if is_cond3 else h
        x_t_for_gates = x_t.detach() if is_cond3 else x_t
        
        # Reset gate: r_t = σ(W_ir x_t + W_hr h_{t-1} + b_ir + b_hr)
        r_t_raw = F.linear(x_t_for_gates, W_ir, b_ir) + F.linear(h_for_gates, W_hr, b_hr)
        # Apply output-domain noise to reset gate BEFORE sigmoid
        if self.enable_output_noise:
            r_t_raw = self._apply_output_noise(r_t_raw, t=t, seed=seed)
        r_t = torch.sigmoid(r_t_raw)
        # Note: For cond3, gradients are already blocked by detaching x_t_for_gates and h_for_gates
        
        # Update gate: z_t = σ(W_iz x_t + W_hz h_{t-1} + b_iz + b_hz)
        z_t_raw = F.linear(x_t_for_gates, W_iz, b_iz) + F.linear(h_for_gates, W_hz, b_hz)
        # Apply output-domain noise to update gate BEFORE sigmoid
        if self.enable_output_noise:
            z_t_raw = self._apply_output_noise(z_t_raw, t=t, seed=seed)
        z_t = torch.sigmoid(z_t_raw)
        # Note: For cond3, gradients are already blocked by detaching x_t_for_gates and h_for_gates
        
        # Candidate gate: n_t = tanh(W_in x_t + r_t ⊙ (W_hn h_{t-1} + b_hn) + b_in)
        # Use h_for_gates and x_t_for_gates (already detached if cond3) to prevent gradients to W_hn and W_in
        h_hn_part = F.linear(h_for_gates, W_hn, b_hn)
        n_t_raw = F.linear(x_t_for_gates, W_in, b_in) + r_t * h_hn_part
        # Apply output-domain noise to candidate gate BEFORE tanh
        if self.enable_output_noise:
            n_t_raw = self._apply_output_noise(n_t_raw, t=t, seed=seed)
        n_t = torch.tanh(n_t_raw)
        # Note: For cond3, gradients are already blocked by detaching x_t_for_gates and h_for_gates
        
        # Update hidden state: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
        # IMPORTANT: For cond3, use detached h to prevent gradients flowing back to previous time step's GRU weights
        # This completely blocks gradients to all GRU weights (W_ir, W_iz, W_in, W_hr, W_hz, W_hn)
        # Note: This also blocks temporal gradients, but that's necessary to block GRU weight gradients
        h_t = (1 - z_t) * n_t + z_t * h_for_gates
        
        return h_t
    
    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        t: int = 0,
        seed: Optional[int] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with memristor non-idealities.
        
        Args:
            input: Input tensor [batch, seq_len, input_size] or [seq_len, batch, input_size]
            hx: Initial hidden state [num_layers * num_directions, batch, hidden_size]
            t: Time/cycle index for drift
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (output, hidden); output may be PackedSequence if input was PackedSequence.
        """
        packed_input = isinstance(input, PackedSequence)
        if packed_input:
            input_seq, lengths = pad_packed_sequence(
                input, batch_first=self.batch_first
            )
            # lengths is on same device as input; keep for repacking
            lengths_cpu = lengths.cpu()
        else:
            input_seq = input
            lengths_cpu = None

        # Handle batch_first
        if self.batch_first:
            batch_size, seq_len, _ = input_seq.shape
        else:
            seq_len, batch_size, _ = input_seq.shape
            input_seq = input_seq.transpose(0, 1)  # [batch, seq_len, input_size]

        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            hx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=input_seq.device,
                dtype=input_seq.dtype,
            )
        
        # Process each layer
        output = input_seq
        hidden_states = []
        
        for layer in range(self.num_layers):
            layer_outputs = []
            layer_hidden = []
            
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weight_ih_name = f'weight_ih_l{layer}{suffix}'
                weight_hh_name = f'weight_hh_l{layer}{suffix}'
                
                # Get weights and apply weight-domain noise
                weight_ih = getattr(self, weight_ih_name)
                weight_hh = getattr(self, weight_hh_name)
                
                seed_offset = (layer * num_directions + direction) * 1000
                weight_seed = seed + seed_offset if seed is not None else None
                
                weight_ih_noisy = self._apply_weight_noise(weight_ih, t=t, seed=weight_seed)
                weight_hh_noisy = self._apply_weight_noise(weight_hh, t=t, seed=weight_seed + 1 if weight_seed is not None else None)
                
                # IMPORTANT: For cond3 (ADC quantization), detach weights to completely block gradients
                # This ensures that gradients cannot flow back to GRU weights even if inputs are detached
                synthetic_noise_type = getattr(self.device_model, 'synthetic_noise_type', 'none')
                is_cond3 = (self.enable_output_noise and synthetic_noise_type == 'cond3_adc_direct')
                if is_cond3:
                    weight_ih_noisy = weight_ih_noisy.detach()
                    weight_hh_noisy = weight_hh_noisy.detach()
                
                # Get biases
                bias_ih = None
                bias_hh = None
                if self.bias:
                    bias_ih_name = f'bias_ih_l{layer}{suffix}'
                    bias_hh_name = f'bias_hh_l{layer}{suffix}'
                    bias_ih = getattr(self, bias_ih_name)
                    bias_hh = getattr(self, bias_hh_name)
                
                # Get initial hidden state for this layer/direction
                hidden_idx = layer * num_directions + direction
                h = hx[hidden_idx]
                
                # Process sequence step by step (mask padding so last hidden is at real end)
                for step in range(seq_len):
                    x_t = output[:, step, :]  # [batch, input_size]
                    
                    # GRU cell forward
                    h_new = self._gru_cell_forward(
                        x_t, h, weight_ih_noisy, weight_hh_noisy,
                        bias_ih, bias_hh, t=t, seed=weight_seed + 2 if weight_seed is not None else None
                    )
                    # Only update hidden for steps within actual length (avoid padding corrupting last hidden)
                    if lengths_cpu is not None:
                        mask = (lengths_cpu > step).to(device=h.device).view(batch_size, 1).expand_as(h)
                        h = torch.where(mask, h_new, h)
                    else:
                        h = h_new
                    
                    layer_outputs.append(h)
                
                # Stack outputs: [batch, seq_len, hidden_size]
                layer_output = torch.stack(layer_outputs, dim=1)
                layer_hidden.append(h)
                output = layer_output
            
            # Concatenate bidirectional outputs
            if self.bidirectional:
                output = torch.cat([layer_output, layer_output], dim=-1)
                hidden_states.extend(layer_hidden)
            else:
                hidden_states.append(layer_hidden[0])
        
        # Convert back to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)  # [seq_len, batch, hidden_size]

        # If input was PackedSequence, pack output for consistency
        if packed_input and lengths_cpu is not None:
            output = pack_padded_sequence(
                output, lengths_cpu, batch_first=self.batch_first, enforce_sorted=False
            )

        # Stack hidden states
        hidden = torch.stack(hidden_states, dim=0)

        return output, hidden
