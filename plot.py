import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义数据
data_types = ['int8', 'int16', 'int32']
datasets = [
    {
        "title": "int8数据类型的三种算子的加速比",
        "operators": ["卷积", "池化", "全连接 (gemm)"],
        "cpu_cycles": [37603, 21361, 21996],
        "accelerator_cycles": [72, 12, 15],
        "speedup_ratio": [522.3, 1780.1, 1466.4]
    },
    {
        "title": "int16数据类型的三种算子的加速比",
        "operators": ["卷积", "池化", "全连接 (gemm)"],
        "cpu_cycles": [37983, 21600, 22321],
        "accelerator_cycles": [72, 12, 15],
        "speedup_ratio": [527.5, 1800.0, 1488.1]
    },
    {
        "title": "int32数据类型的三种算子的加速比",
        "operators": ["卷积", "池化", "全连接 (gemm)"],
        "cpu_cycles": [37539, 21238, 21721],
        "accelerator_cycles": [72, 12, 15],
        "speedup_ratio": [521.4, 1769.8, 1448.1]
    }
]

# 为每个数据类型绘制一张图
for i, dataset in enumerate(datasets):
    operators = dataset["operators"]
    cpu_cycles = dataset["cpu_cycles"]
    accelerator_cycles = dataset["accelerator_cycles"]
    speedup_ratio = dataset["speedup_ratio"]

    # 计算柱子的位置
    x = np.arange(len(operators))  # [0, 1, 2]
    width = 0.25  # 柱子宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx() # 创建第二个Y轴用于加速比

    # 绘制CPU和加速器的柱子（使用第一个Y轴，并设为对数刻度）
    bars1 = ax.bar(x - width/2 - 0.02, cpu_cycles, width * 0.45, label='CPU运行时钟周期', color='skyblue')
    bars2 = ax.bar(x + width/2 + 0.02, accelerator_cycles, width * 0.45, label='加速器运行时钟周期', color='lightcoral')
    ax.set_yscale('log')  # 关键：设置第一个Y轴为对数刻度
    ax.set_ylabel('时钟周期数 (Log Scale)', color='black')

    # 绘制加速比的柱子（使用第二个Y轴，线性刻度）
    bars3 = ax2.bar(x + width, speedup_ratio, width * 0.45, label='加速比', color='lightgreen', alpha=0.7)
    ax2.set_ylabel('加速比', color='black')

    # 设置坐标轴标签和标题
    ax.set_xlabel('算子类型')
    ax.set_title(dataset["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(operators)

    # 设置图例
    # 将两个轴上的图例合并
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 添加数值标签 (注意对数坐标下的高度处理)
    def add_log_value_labels(bars, axis, is_speedup=False):
        for bar in bars:
            height = bar.get_height()
            # 在对数坐标系中，文本的位置计算稍微复杂一点，这里简化处理
            # 使用bar.get_y() + height来获取柱顶的实际Y值（对数坐标）
            y_pos = bar.get_y() + height 
            # 为了让标签在柱子顶部可见，稍微向上偏移一点
            offset_factor = 1.05 # 对数尺度下，乘以一个小的因子来模拟"上方"
            display_y = y_pos * offset_factor
            
            text_val = f'{height:.1f}' if height != int(height) else f'{int(height)}'
            
            axis.text(bar.get_x() + bar.get_width() / 2., display_y,
                      text_val,
                      ha='center', va='bottom', fontsize=9)

    add_log_value_labels(bars1, ax)
    add_log_value_labels(bars2, ax)
    # 加速比的标签用第二个轴 (ax2)，它仍然是线性的
    add_log_value_labels(bars3, ax2, is_speedup=True)


    # 调整布局防止标签被截断
    fig.tight_layout()

    # 显示图形
    plt.show()