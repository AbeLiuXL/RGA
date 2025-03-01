import os
import torch
import numpy as np
from utils.utils import seed_torch
seed_torch()

from PIL import Image
import cv2
from torchvision import transforms as T
import argparse
from path import Path
from multiprocessing import Process

from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.build_sam import sam_model_registry

from utils.get_rgm import segment_and_dilate
from attack.attacker import ATTACK_SAM

def single_img_attack(img_path,sam,mask_generator,attack_sam,args):
    print(img_path)
    # Read the image and convert it to RGB format
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Get Region-Guided Map
    rgm = segment_and_dilate(mask_generator=mask_generator, img=img, grid_rate=0.1, dilate_iter=50)
    rgm = Image.fromarray(rgm).convert("RGB")

    save_dir_rgm = args.save_dir / 'rgm_imgs'
    if not save_dir_rgm.exists():
        save_dir_rgm.mkdir()
    rgm.save(save_dir_rgm/f'{Path(img_path).stem}_rgm.png')

    pil_image = Image.fromarray(img).convert("RGB")
    oldh = pil_image.size[1]
    oldw = pil_image.size[0]
    transform_resize = ResizeLongestSide(sam.image_encoder.img_size)
    new_size = transform_resize.get_preprocess_shape(oldh, oldw, 1024)
    transforms = T.Compose([
        T.Resize(new_size),
        T.ToTensor()
    ])
    input_image_torch = transforms(pil_image).unsqueeze(0).cuda()
    tar_img_torch = transforms(rgm).unsqueeze(0).cuda()
    adv_image_torch = attack_sam(input_image_torch, tar_img=tar_img_torch)
    adv_image = T.ToPILImage()(T.Resize((oldh, oldw))(adv_image_torch[0]))
    save_dir_adv= args.save_dir / 'adv_imgs'
    if not save_dir_adv.exists():
        save_dir_adv.mkdir()
    adv_image.save(save_dir_adv/f'{Path(img_path).stem}_adv.png')

def multi_img_attack(args,gpu_id):
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth").cuda().eval()
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=0.88)
    attack_sam = ATTACK_SAM(model=sam, args=args)
    for i in range(args.length):
        print(f'{i}/{args.length}')
        img_id = args.min_item + i
        img_path = args.data_dir / f'sa_{img_id}.jpg'
        single_img_attack(img_path,sam,mask_generator,attack_sam,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/sa_1_10')
    parser.add_argument('--img_path', type=str, default='imgs/dog.jpg')
    parser.add_argument('--is_single_img', type=int, default=1, choices=[0, 1])
    parser.add_argument('--data_len', type=int, default=10)
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument("--gpu_id", nargs='+', type=str, default="0")

    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--num_iter", type=int, default=40)
    parser.add_argument("--scale", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--momentum", type=float, default=0.4)
    parser.add_argument("--resize_rate", type=float, default=0.9)
    parser.add_argument("--diversity_prob", type=float, default=0.7)
    parser.add_argument("--gamma", type=int, default=3)
    # parser.add_argument("--norm", type=str, default='l1')
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--is_ti", type=int, default=0, choices=[0, 1])
    parser.add_argument("--is_dim", type=int, default=0, choices=[0, 1])
    parser.add_argument("--is_target", type=int, default=1, choices=[0, 1])
    parser.add_argument("--print_iter", type=int, default=10)

    args = parser.parse_args()
    args = parser.parse_args()
    args.save_dir = Path('output') / args.run_name
    args.epsilon = args.epsilon / 255.0
    print("save_dir:",args.save_dir)
    if not args.save_dir.exists():
        args.save_dir.mkdir()
    if args.is_single_img:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id[0]
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth").cuda().eval()
        mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=0.88)
        attack_sam = ATTACK_SAM(model=sam, args=args)
        single_img_attack(args.img_path,sam,mask_generator,attack_sam,args)
    else:
        gpu_len = len(args.gpu_id)
        args.data_dir = Path(args.data_dir)
        if gpu_len > 1:
            L = args.data_len // gpu_len
            for i, gpu_id in enumerate(args.gpu_id):
                args.min_item = i * L + 1
                if i == (gpu_len - 1):
                    args.length = L + args.data_len % gpu_len
                else:
                    args.length = L
                print("gpu_id_{}_length_{}".format(gpu_id, args.length))
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                Process(target=multi_img_attack, args=(args, gpu_id,)).start()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id[0]
            args.min_item = 1
            args.length = args.data_len
            multi_img_attack(args, args.gpu_id[0])

