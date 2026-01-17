"""
Experiment orchestration and visualization modules.
"""

# 使用延迟导入避免 RuntimeWarning
# 当使用 python -m src.experiments.run_experiment 时，如果 __init__.py 中直接导入 run_experiment
# 会导致 run_experiment 在 sys.modules 中，然后 runpy 会发出警告
# 解决方案：使用 __getattr__ 实现延迟导入（Python 3.7+）

__all__ = ["run_experiment", "plot_accuracy_curve", "plot_parameter_sweep"]


def __getattr__(name):
    """延迟导入模块以避免 RuntimeWarning"""
    if name == "run_experiment":
        from .run_experiment import run_experiment
        return run_experiment
    elif name == "plot_accuracy_curve":
        from .visualize import plot_accuracy_curve
        return plot_accuracy_curve
    elif name == "plot_parameter_sweep":
        from .visualize import plot_parameter_sweep
        return plot_parameter_sweep
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


