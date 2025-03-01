from utils.utils import seed_torch
seed_torch()
from segment_anything.build_sam import sam_model_registry
from PIL import Image
from torchvision import transforms as T
import numpy as np
import torch
import cv2
import json
from pycocotools.mask import decode as pycocotools_decode
from segment_anything.predictor import SamPredictor
from utils.utils import IoU,sample_pixel_in_mask,calculate_precision_recall_f1
from path import Path
import os
import argparse
from tqdm import trange
from FastSAM.fastsam import FastSAM, FastSAMPrompt

from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam

from MobileSAM.mobile_sam import  SamPredictor as MobiSamPredictor
from MobileSAM.mobile_sam import sam_model_registry as mobile_sam_model_registry
def obtain_single_image_evaluation_config(
        labelpath:str,
        eval_point_sample_num: int = 5,
        eval_box_sample_num: int = 3,
        eval_box_sample_rescale_low: float = 0.9,
        eval_box_sample_rescale_high: float = 1.1,
        mask_num_limit: int = 10
):
    # clean_cv2_image = cv2.imread(imgpath)
    # clean_cv2_image = cv2.cvtColor(clean_cv2_image, cv2.COLOR_BGR2RGB)
    mask_list = []  # list of mask: M-sequence of [H, W]-ndarray
    prompt_list = []  # list of list of prompt: M lists, each list containing (P+B)-sequence of [2 or 4]-ndarray
    with open(labelpath, 'r') as f:
        label_data = json.load(f)
        gt_masks = label_data['annotations']
        gt_masks = gt_masks[:mask_num_limit]
        for mask_id, gt_mask in enumerate(gt_masks):  # from 0 to M-1
            gt_segmentation = pycocotools_decode(gt_mask['segmentation'])
            gt_height = gt_mask['segmentation']['size'][0]
            gt_width = gt_mask['segmentation']['size'][1]
            gt_bbox = gt_mask['bbox']
            prompt_list_for_current_mask = []
            mask_list.append(gt_segmentation)
            # sample P point prompts per mask
            points = sample_pixel_in_mask(gt_segmentation, sample_num=eval_point_sample_num)
            for point_prompt in points:
                prompt_list_for_current_mask.append(point_prompt)
            # sample B box prompts per mask
            cx = gt_bbox[0] + gt_bbox[2] / 2
            cy = gt_bbox[1] + gt_bbox[3] / 2
            w = gt_bbox[2]
            h = gt_bbox[3]
            for box_prompt_id in range(eval_box_sample_num):
                randw = np.random.randint(int(eval_box_sample_rescale_low * w), int(eval_box_sample_rescale_high * w))
                randh = np.random.randint(int(eval_box_sample_rescale_low * h), int(eval_box_sample_rescale_high * h))
                rand_xl = max(int(cx - randw / 2), 0)
                rand_xr = min(int(cx + randw / 2), gt_width - 1)
                rand_yu = max(int(cy - randh / 2), 0)
                rand_yd = min(int(cy + randh / 2), gt_height - 1)
                rand_bbox = np.array([rand_xl, rand_yu, rand_xr, rand_yd])
                prompt_list_for_current_mask.append(rand_bbox)
            prompt_list.append(prompt_list_for_current_mask)
    return mask_list, prompt_list

# assume the adversarial input has already been set
@torch.no_grad()
def sam_single_image_batched_prompt_test(
    sam_model: SamPredictor,
    gt_mask_list: list,
    prompt_list: list,
):
    # current implementation only supports single image input + single prompt input
    assert len(gt_mask_list) == len(prompt_list)
    iou_list = []
    # t0 = time.time()
    precision_list, recall_list, f1_score_list=[],[],[]
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        #import pdb; pdb.set_trace()
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            if len(prompt) == 2: # point prompt
                point = np.array([prompt])
                label = np.array([1])
                pred_mask, _, _ = sam_model.predict(point, label, None, multimask_output=False, return_logits=False)
            elif len(prompt) == 4: # box prompt
                box = np.array(prompt)
                pred_mask, _, _ = sam_model.predict(None, None, box, multimask_output=False, return_logits=False)
            # print(pred_mask[0].shape,gt_mask.shape)
            pred_mask_iou = IoU(pred_mask[0], gt_mask)
            precision, recall, f1_score = calculate_precision_recall_f1(pred_mask[0], gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        # print("time:", time.time() - t0)
    return iou_list,precision_list,recall_list,f1_score_list
import time
@torch.no_grad()
def fastsam_single_image_batched_prompt_test(
    fastsam_model: FastSAM,
    adv_img_cv2: np.ndarray,
    gt_mask_list: list,
    prompt_list: list,
):
    iou_list = []
    h, w = adv_img_cv2.shape[:2]
    imgsz=1024
    # imgsz = max(h, w)
    # if imgsz%32!=0:
    #     tmp=imgsz//32
    #     imgsz = 32*(tmp+1)

    everything_results = fastsam_model(adv_img_cv2, retina_masks=True, imgsz=imgsz, conf=0.4, iou=0.9)
    prompt_process = FastSAMPrompt(adv_img_cv2, everything_results)
    t0=time.time()
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            if len(prompt) == 2:
                point = np.array([prompt])
                label = np.array([1])
                pred_mask = prompt_process.point_prompt(points=point, pointlabel=label)
            elif len(prompt) == 4:
                pred_mask = prompt_process.box_prompt(bbox=prompt.tolist())
            if len(pred_mask) == 0: # no mask predicted
                current_mask_iou_list.append(0.0)
                continue
            pred_mask_iou = IoU(pred_mask[0], gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
        print("time:",time.time()-t0)
    return iou_list
@torch.no_grad()
def efficientsam_single_image_batched_prompt_test(efficientsam,adv_img, gt_mask_list, prompt_list):
    # current implementation only supports single image input + single prompt input
    assert len(gt_mask_list) == len(prompt_list)
    iou_list = []
    img_tensor = T.ToTensor()(adv_img).cuda()
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        # import pdb; pdb.set_trace()
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            # print(prompt)
            pts_sampled = torch.reshape(torch.tensor(prompt), [1, 1, -1, 2]).cuda()
            if len(prompt) == 2:
                pts_labels = torch.reshape(torch.tensor([1]), [1, 1, -1]).cuda()
            else:
                pts_labels = torch.reshape(torch.tensor([2,3]), [1, 1, -1]).cuda()
            predicted_logits, _ = efficientsam(
                img_tensor[None, ...],
                pts_sampled,
                pts_labels,
            )
            pred_mask_iou = IoU(predicted_logits[0], gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
    return iou_list

@torch.no_grad()
def efficientsam_single_image_batched_prompt_test(efficientsam,adv_img, gt_mask_list, prompt_list):
    # current implementation only supports single image input + single prompt input
    assert len(gt_mask_list) == len(prompt_list)
    iou_list = []
    img_tensor = T.ToTensor()(adv_img).cuda()
    for gt_mask_id, (gt_mask, prompt_current_mask_list) in enumerate(zip(gt_mask_list, prompt_list)):
        current_mask_iou_list = []
        # import pdb; pdb.set_trace()
        for prompt_id, prompt in enumerate(prompt_current_mask_list):
            # print(prompt)
            pts_sampled = torch.reshape(torch.tensor(prompt), [1, 1, -1, 2]).cuda()
            if len(prompt) == 2:
                pts_labels = torch.reshape(torch.tensor([1]), [1, 1, -1]).cuda()
            else:
                pts_labels = torch.reshape(torch.tensor([2,3]), [1, 1, -1]).cuda()
            predicted_logits, predicted_iou = efficientsam(
                img_tensor[None, ...],
                pts_sampled,
                pts_labels,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            # predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            # print(predicted_logits.shape,gt_mask.shape)
            pred_mask_iou = IoU(predicted_logits, gt_mask)
            current_mask_iou_list.append(pred_mask_iou)
        iou_list.append(current_mask_iou_list)
    return iou_list

def main(args):
    if args.attack_model=="vit_b":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
        sam = sam.cuda().eval()
        saml_eval = SamPredictor(sam)
    elif args.attack_model=="vit_l":
        sam = sam_model_registry["vit_l"](checkpoint="ckpt/sam_vit_l_0b3195.pth")
        sam = sam.cuda().eval()
        saml_eval = SamPredictor(sam)
    elif args.attack_model == "vit_h":
        sam = sam_model_registry["vit_h"](checkpoint="ckpt/sam_vit_h_4b8939.pth")
        sam = sam.cuda().eval()
        saml_eval = SamPredictor(sam)
    elif args.attack_model == "fastsam":
        saml_eval = FastSAM("ckpt/FastSAM-x.pt")
        saml_eval.model.cuda().eval()
    elif args.attack_model == "mobilesam":
        sam = mobile_sam_model_registry["vit_t"]("MobileSAM/weights/mobile_sam.pt").cuda().eval()
        saml_eval = MobiSamPredictor(sam)
    elif args.attack_model == "efficientsam":
        saml_eval = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="EfficientSAM/weights/efficient_sam_vitt.pt",
    ).cuda().eval()
    else:
        raise ValueError(f"Unsupported attack model type: {args.attack_model}")
    iou_list_all=[]
    precision_list_all, recall_list_all, f1_score_list_all=[],[],[]
    if Path(args.adv_dir).files('*.png'):
        extension = 'png'
    else:
        extension = 'jpg'
    for i in trange(args.data_len):
        adv_img_path = Path(args.adv_dir)/f'sa_{i+1}_adv.{extension}'
        label_path = Path(args.data_dir)/f'sa_{i+1}.json'
        adv_img = cv2.imread(adv_img_path)
        adv_img = cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB)
        gt_mask_list, prompt_list = obtain_single_image_evaluation_config(label_path)
        if args.attack_model == "fastsam":
            iou_list = fastsam_single_image_batched_prompt_test(saml_eval,adv_img, gt_mask_list, prompt_list)
            iou_list_all.extend(np.array(iou_list).flatten())
        elif args.attack_model == "efficientsam":
            iou_list = efficientsam_single_image_batched_prompt_test(saml_eval,adv_img, gt_mask_list, prompt_list)
            iou_list_all.extend(np.array(iou_list).flatten())
        else:
            saml_eval.set_image(adv_img)
            iou_list,precision_list,recall_list,f1_score_list = sam_single_image_batched_prompt_test(saml_eval, gt_mask_list, prompt_list)
            iou_list_all.extend(np.array(iou_list).flatten())
            precision_list_all.extend(np.array(precision_list).flatten())
            recall_list_all.extend(np.array(recall_list).flatten())
            f1_score_list_all.extend(np.array(f1_score_list).flatten())
    flattened_array = np.array(iou_list_all)
    iou_mean = np.mean(flattened_array)
    iou_std = np.std(flattened_array)
    asr_50 = np.sum(flattened_array <= 50) / len(flattened_array) * 100
    asr_10 = np.sum(flattened_array <= 10) / len(flattened_array) * 100
    print(f"{args.adv_dir}-{args.attack_model},mIoU:{iou_mean:.2f}+-{iou_std:.2f} - ASR@50: {asr_50:.2f} - ASR@10: {asr_10:.2f}")
    print(f"& {iou_mean:.2f}+-{iou_std:.2f} & {asr_50:.2f} & {asr_10:.2f}")
    # if args.attack_model != "fastsam":
    #     print(f'precision：{np.array(precision_list_all).mean()*100:.2f},recall:{np.array(recall_list_all).mean()*100:.2f},f1_score:{np.array(f1_score_list_all).mean()*100:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/sa_1_10')
    parser.add_argument('--data_len', type=int, default=-1)
    parser.add_argument('--adv_dir', type=str, default='test')
    parser.add_argument("--gpu_id", nargs='+', type=str, default="0")
    parser.add_argument("--test", type=int, default=0,choices=[0,1])
    parser.add_argument('--attack_model', type=str, default='vit_b',choices=['vit_b','vit_l','vit_h','fastsam','efficientsam','mobilesam'])
    args = parser.parse_args()
    if args.test:
        imgpath = 'checkpoints/pgd_img.png'
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_mask_list, prompt_list = obtain_single_image_evaluation_config('checkpoints/imgs/sa_4.json')
        sam = sam_model_registry["vit_h"](checkpoint="ckpt/sam_vit_h_4b8939.pth").cuda()
        sam.eval()
        saml_eval = SamPredictor(sam)
        saml_eval.set_image(img)
        iou_list = sam_single_image_batched_prompt_test(saml_eval, gt_mask_list, prompt_list)
        print(iou_list)
        flattened_array = np.array(iou_list).flatten()
        iou_mean = np.mean(flattened_array)
        iou_std = np.std(flattened_array)
        asr_50 = np.sum(flattened_array <= 50) / len(flattened_array) * 100
        asr_10 = np.sum(flattened_array <= 10) / len(flattened_array) * 100
        print(f"\n{iou_mean:.2f}+-{iou_std:.2f} - ASR@50: {asr_50:.2f} - ASR@10: {asr_10:.2f}")
    else:
        if args.data_len<=0:
            if Path(args.adv_dir).files('*.png'):
                extension = 'png'
            else:
                extension = 'jpg'
            args.data_len = len(Path(args.adv_dir).files(f'*.{extension}'))
            print("data_len:",args.data_len)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id[0]
        main(args)

