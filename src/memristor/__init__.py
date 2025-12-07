"""
Memristor device modeling and compensation modules.
"""

from .device_model import MemristorDeviceModel
from .mapping import (
    map_weights_linear,
    map_weights_log,
    differential_pair_mapping,
    reshape_conv_to_matrix,
)
from .compensation import (
    hardware_aware_training,
    # learned_mapping_train,  # OLD - replaced by train_weight_mapping
    # post_train_learned_mapping,
    joint_hat_mapping_train,
)
from .energy_estimator import EnergyEstimator

__all__ = [
    "MemristorDeviceModel",
    "map_weights_linear",
    "map_weights_log",
    "differential_pair_mapping",
    "reshape_conv_to_matrix",
    "hardware_aware_training",
    # "learned_mapping_train",
    # "post_train_learned_mapping",
    "joint_hat_mapping_train",
    "EnergyEstimator",
]


