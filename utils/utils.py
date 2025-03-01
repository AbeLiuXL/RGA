import numpy as np
import torch
import random
import os
def IoU(mask1, mask2):
    if mask1.__class__ == torch.Tensor:
        mask1 = mask1.detach().cpu().numpy()
    if mask2.__class__ == torch.Tensor:
        mask2 = mask2.detach().cpu().numpy()
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou= 100 * intersection / union
    # 计算Dice系数
    dice = (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) != 0 else 0
    return iou


def calculate_precision_recall_f1(mask, gt_mask):
    if mask.__class__ == torch.Tensor:
        mask = mask.detach().cpu().numpy()
    if gt_mask.__class__ == torch.Tensor:
        gt_mask = gt_mask.detach().cpu().numpy()
    # 确保mask和gt_mask都是二进制的，即包含0和1
    mask = mask.astype(np.bool_)
    gt_mask = gt_mask.astype(np.bool_)

    # 计算True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = np.logical_and(mask, gt_mask).sum()  # mask和gt_mask都是1的像素
    FP = np.logical_and(mask, np.logical_not(gt_mask)).sum()  # mask是1但gt_mask是0的像素
    FN = np.logical_and(np.logical_not(mask), gt_mask).sum()  # mask是0但gt_mask是1的像素

    # 计算Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # 计算Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def sample_pixel_in_mask(mask, sample_num=10):
    # mask is a numpy 2d-array of shape [H, W]
    # output P point prompts in a list of shape [P, 2]
    inmask_pixel_positions = np.flip(np.argwhere(mask == True), axis=1)
    sample_size = min(sample_num, inmask_pixel_positions.shape[0])
    sampled_pixel_id = np.random.choice(inmask_pixel_positions.shape[0], size=sample_size, replace=False)
    sampled_pixel_pos = inmask_pixel_positions[sampled_pixel_id]
    return sampled_pixel_pos

def seed_torch(seed=3407):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法