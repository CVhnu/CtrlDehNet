# from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from basicsr.archs import build_network
from collections import Counter
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

haze_path = '/home/lhp/Desktop/HDP_test'
clear_path = '/home/lhp/code/RIDCP_dehazing/results-Nohdp'

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

dict_total = []
if __name__ == '__main__':
    ckpt_path = '/home/lhp/code/RIDCP_dehazing/experiments/ridcp+fourier/models/net_g_30000.pth'
    # print(hq_opt['codebook_params'].size())
    net_vq = VQWeightDehazeNet(**hq_opt).cuda()
    net_vq.load_state_dict(torch.load(ckpt_path)['params'])

    i = 0
    for filename in os.listdir(haze_path):
        filepath = os.path.join(haze_path, filename)
        image = cv2.imread(filepath)[:, :, ::-1] / 255.0
        image = torch.FloatTensor(image).unsqueeze(0).cuda().permute(0, 3, 1, 2)

        _, index_list = net_vq.test(image)
        dict_total += list(index_list[0].flatten(0).cpu().numpy())

        i = i +1
        print(i)
        if i > 100:
            break

result = Counter(dict_total)
result_image = np.zeros((32, 32))
print(len(result))
for k in result.keys():
    k_index = int(k)
    result_image[k // 32, k % 32] = result[k]

# result_image = (result_image - result_image.min()) / (result_image.max() - result_image.min())
# result_image = np.log(result_image + 1)
plt.imshow(result_image)
plt.savefig('visual_code_dehaze_exp.png')