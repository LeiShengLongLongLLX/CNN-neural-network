# 终端打印模型权重参数信息
import torch


# 参数配置
# 模型权重文件路径
weight_file = r"G:\Project\CNN_accerator_basedonRISC-V\ModelWeigh\MobileNetV1_svhn.pth"

# 是否打印完整张量内容（True 打印全部，False 只打印均值、方差）
print_full_tensor = False

# ===========================
# 1. 读取模型权重
# ===========================
print(f"\n[INFO] Loading weights from:\n{weight_file}\n")
state_dict = torch.load(weight_file, map_location="cpu")

print("[INFO] Model weights loaded successfully!\n")

# ===========================
# 2. 遍历所有参数张量
# ===========================
for name, tensor in state_dict.items():
    print(f"Param Name : {name}")
    print(f"Shape      : {tuple(tensor.shape)}")

    if print_full_tensor:
        print("Tensor:\n", tensor)
    else:
        # 输出统计信息
        print(f"Mean       : {tensor.float().mean().item():.6f}")
        print(f"Std        : {tensor.float().std().item():.6f}")
        print(f"Min / Max  : {tensor.float().min().item():.6f} / {tensor.float().max().item():.6f}")

    print("-" * 60)
