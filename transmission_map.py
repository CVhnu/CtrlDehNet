import torch
import matplotlib.pyplot as plt


def compute_transmission_map(input_tensor, omega=0.95, epsilon=1e-6, show_plot=False):
    """
    计算透射图（移除最小滤波以提升速度）

    参数:
        input_tensor: 输入图像Tensor, shape (1, 3, H, W)或(3, H, W), 值范围[0,1]或[0,255]
        omega: 透射调节参数
        epsilon: 防除零小常数
        show_plot: 是否显示可视化

    返回:
        transmission_map: 透射图Tensor, shape (1, 1, H, W)
    """
    # 处理输入维度
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # 检查输入形状
    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 3, "输入应为(1, 3, H, W)或(3, H, W)"

    # 归一化到[0,1]
    if input_tensor.max() > 1.0:
        input_tensor.div_(255.0)

    # 计算暗通道（仅通道最小值，不移除最小滤波）
    dark_channel = torch.min(input_tensor, dim=1, keepdim=True).values  # (1, 1, H, W)

    # 估计大气光
    flat_dark = dark_channel.view(-1)
    top_k = max(1, int(flat_dark.numel() * 0.001))
    atmos_light = flat_dark.topk(top_k).values.mean()

    # 计算透射图
    transmission_map = 1 - omega * (dark_channel / (atmos_light + epsilon))
    transmission_map.clamp_(0.2, 1.0)

    # 可视化
    if show_plot:
        input_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        trans_np = transmission_map.squeeze().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_np)
        plt.title("Input Image")

        plt.subplot(1, 2, 2)
        plt.imshow(trans_np, cmap='gray', vmin=0.1, vmax=1.0)
        plt.colorbar(label='Transmission')
        plt.title("Transmission Map (Without Min Filter)")
        plt.tight_layout()
        plt.show()

    return transmission_map