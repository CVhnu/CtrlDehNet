from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from collections import Counter
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 图像路径设置（3个目录）
clear_path = '/home/lhp/Desktop/clear_activation'
hdpimg_path = '/home/lhp/code/RIDCP_dehazing/results-HDP'
alpha_0path = '/home/lhp/code/RIDCP_dehazing/results-test0000'

# 模型配置参数
hq_opt = {
    'gt_resolution': 256,
    'norm_type': 'gn',
    'act_type': 'silu',
    'scale_factor': 1,
    'codebook_params': [[64, 1024, 512]],
    'LQ_stage': True,
    'use_quantilize': True,
    'use_semantic_loss': False
}


def process_images(image_path, model, max_images=100):
    """处理指定路径下的图像并返回编码索引列表"""
    code_indices = []
    image_count = 0

    for filename in os.listdir(image_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue

        filepath = os.path.join(image_path, filename)
        try:
            image = cv2.imread(filepath)
            if image is None:
                continue

            image = image[:, :, ::-1] / 255.0
            image = torch.FloatTensor(image).unsqueeze(0).cuda().permute(0, 3, 1, 2)

            _, index_list = model.test(image)
            code_indices.extend(list(index_list[0].flatten(0).cpu().numpy()))

            image_count += 1
            print(f"已处理 {image_count} 张图像")
            if image_count >= max_images:
                break
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue

    return code_indices


if __name__ == '__main__':
    # 加载模型
    ckpt_path = '/home/lhp/code/RIDCP_dehazing/experiments/ridcp+fourier/models/net_g_30000.pth'
    net_vq = VQWeightDehazeNet(**hq_opt).cuda()
    net_vq.load_state_dict(torch.load(ckpt_path)['params'])
    net_vq.eval()

    # 处理三类图像
    print("开始处理清晰图像...")
    clear_indices = process_images(clear_path, net_vq, max_images=100)

    print("开始处理HDP图像...")
    hdpimg_indices = process_images(hdpimg_path, net_vq, max_images=100)

    print("开始处理alpha=0图像...")
    alpha0_indices = process_images(alpha_0path, net_vq, max_images=100)

    # 统计编码频率
    total_codes = 1024
    clear_counts = [Counter(clear_indices).get(i, 0) for i in range(total_codes)]
    hdpimg_counts = [Counter(hdpimg_indices).get(i, 0) for i in range(total_codes)]
    alpha0_counts = [Counter(alpha0_indices).get(i, 0) for i in range(total_codes)]

    # 归一化
    clear_total = sum(clear_counts)
    hdpimg_total = sum(hdpimg_counts)
    alpha0_total = sum(alpha0_counts)

    clear_norm = [c / clear_total if clear_total else 0 for c in clear_counts]
    hdpimg_norm = [c / hdpimg_total if hdpimg_total else 0 for c in hdpimg_counts]
    alpha0_norm = [c / alpha0_total if alpha0_total else 0 for c in alpha0_counts]

    # 筛选HDP比alpha=0更接近clear的编码
    # 计算每个编码的HDP与clear的差异，以及alpha=0与clear的差异
    hdp_diff = [abs(hdpimg_norm[i] - clear_norm[i]) for i in range(total_codes)]
    alpha0_diff = [abs(alpha0_norm[i] - clear_norm[i]) for i in range(total_codes)]

    # 找出HDP差异小于alpha=0差异的编码索引
    target_indices = [i for i in range(total_codes) if hdp_diff[i] < alpha0_diff[i]]
    count_target = len(target_indices)
    print(f"HDP类数值比alpha=0类更逼近clear类的编码有 {count_target} 个")

    # 为这些编码准备数据
    clear_target = [clear_norm[i] for i in target_indices]
    hdp_target = [hdpimg_norm[i] for i in target_indices]
    alpha0_target = [alpha0_norm[i] for i in target_indices]

    # 绘图设置
    plt.figure(figsize=(22, 10))
    bar_width = 0.25
    index = np.arange(count_target)

    # 绘制柱状图
    plt.bar(index - bar_width, clear_target, bar_width, label='clear',
            color='#66B2FF', alpha=0.8)
    plt.bar(index, hdp_target, bar_width, label='step2(HDP)',
            color='#99FF99', alpha=0.8)
    plt.bar(index + bar_width, alpha0_target, bar_width, label='step1',
            color='#FFCC99', alpha=0.8)

    # 图表设置
    plt.xlabel('Code_Index', fontsize=18)
    plt.ylabel('Activation_Frequency', fontsize=18)
    # plt.title(f'HDP比alpha=0更接近clear的编码 (共{count_target}个)', fontsize=16)
    plt.legend(fontsize=18)

    # 只显示部分刻度，避免拥挤
    tick_interval = max(1, count_target // 20)  # 最多显示20个刻度
    plt.xticks(index[::tick_interval], [str(target_indices[i]) for i in range(0, count_target, tick_interval)],
               rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(axis='y', linestyle='--',alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig('hdp_closer_to_clear_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()

    print("编码对比图已保存")
