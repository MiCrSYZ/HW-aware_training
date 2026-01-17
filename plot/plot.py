import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter

# 设置中文字体和图表样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('default')

# 定义参数
t = np.logspace(0, np.log10(3000), 1000)  # 1到3000的对数均匀分布点
alphas = [1e-2, 1e-3, 1e-4]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 不同alpha对应的颜色
labels = [r'$\alpha = 10^{-2}$', r'$\alpha = 10^{-3}$', r'$\alpha = 10^{-4}$']

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制第一条曲线 f(t) = t^(-α)
for i, alpha in enumerate(alphas):
    f1 = t ** (-alpha)
    ax1.loglog(t, f1, color=colors[i], linewidth=2.5, label=labels[i])

ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel(r'$f(t) = t^{-\alpha}$', fontsize=12)
ax1.set_title(r'幂函数衰减曲线 $f(t) = t^{-\alpha}$', fontsize=14)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)
ax1.set_xlim(1, 3000)

# 绘制第二条曲线 f(t) = 1 - αlg(1+t)
for i, alpha in enumerate(alphas):
    f2 = 1 - alpha * np.log10(1 + t)
    ax2.semilogx(t, f2, color=colors[i], linewidth=2.5, label=labels[i])

ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel(r'$f(t) = 1 - \alpha \lg(1+t)$', fontsize=12)
ax2.set_title(r'对数衰减曲线 $f(t) = 1 - \alpha \lg(1+t)$', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)
ax2.set_xlim(1, 3000)
ax2.set_ylim(0.9, 1.01)  # 调整y轴范围以更好地显示曲线

# 调整布局
plt.tight_layout()
plt.show()

# 可选：单独显示每个alpha的三组对比
fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, alpha in enumerate(alphas):
    # 计算两个函数的值
    f1 = t ** (-alpha)
    f2 = 1 - alpha * np.log10(1 + t)

    # 绘制对比图
    axes[i].semilogx(t, f1, color='#FF6B6B', linewidth=2.5,
                     label=r'$t^{-' + f'{alpha:.0e}' + '}$')
    axes[i].semilogx(t, f2, color='#4ECDC4', linewidth=2.5,
                     label=r'$1 - ' + f'{alpha:.0e}' + r'\lg(1+t)$')

    axes[i].set_xlabel('t', fontsize=10)
    axes[i].set_ylabel('f(t)', fontsize=10)
    axes[i].set_title(f'α = {alpha}', fontsize=12)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=9)
    axes[i].set_xlim(1, 3000)

plt.tight_layout()
plt.show()