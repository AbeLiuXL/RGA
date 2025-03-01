import torch
import torch.nn as nn
from .attack import Attack
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
from torch.nn import CosineSimilarity
class ATTACK_SAM(Attack):
    def __init__(self,model,args,gpu_id='0'):
        super(ATTACK_SAM, self).__init__('ATTACK_SAM',args)
        """
        self.args.resize_rate = 0.9
        self.args.diversity_prob = 0.7
        self.args.momentum=1.0
        """
        self.model = model
        self.kernel_name = 'gaussian'
        self.gpu_id = gpu_id
        self.nsig = 3
        self.len_kernel = 15
        self.select_feature_layers = [4, 5, 6, 7]
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.args.resize_rate)

        if self.args.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        return padded if torch.rand(1) < self.args.diversity_prob else x
    def rst(self,before_pasted, beta=0.01):
        if beta>0:
            B, C, H, W = before_pasted.size()
            translate_x = torch.FloatTensor(torch.Size([B])).uniform_(-beta, beta).type_as(before_pasted)
            translate_y = torch.FloatTensor(torch.Size([B])).uniform_(-beta, beta).type_as(before_pasted)
            rotate_ = torch.FloatTensor(torch.Size([B])).uniform_(-beta * 90 / 180 * np.pi,
                                                                  beta * 90 / 180 * np.pi).type_as(
                before_pasted)
            scale_ = torch.FloatTensor(torch.Size([B])).uniform_(1 - beta, 1 + beta).type_as(before_pasted)

            theta = torch.zeros([B, 2, 3]).type_as(before_pasted)
            theta[:, 0, 0] = scale_ * torch.cos(rotate_)
            theta[:, 0, 1] = torch.sin(rotate_)
            theta[:, 0, 2] = translate_x
            theta[:, 1, 0] = -torch.sin(rotate_)
            theta[:, 1, 1] = scale_ * torch.cos(rotate_)
            theta[:, 1, 2] = translate_y
            grid = F.affine_grid(theta, size=torch.Size([B, C, H, W]), align_corners=False).type_as(before_pasted)
            before_pasted_aug = F.grid_sample(before_pasted, grid, align_corners=False)
            return before_pasted_aug
        else:
            return before_pasted

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def cal_cos_loss(self, attack_img, src_img_feature, tar_img_feature=None):
        self.model.eval()
        attack_img_feature = self.model.image_encoder(attack_img).reshape(1, -1)
        loss =  CosineSimilarity()(attack_img_feature, src_img_feature)
        # loss = torch.nn.SmoothL1Loss()(emb_attack_img,emb_target_img)#50000
        if tar_img_feature != None:
            loss = loss - CosineSimilarity()(attack_img_feature, tar_img_feature)
        return loss.mean()

    def forward(self,src_img,tar_img=None):
        self.model.eval()
        with torch.set_grad_enabled(False):
            src_img_norm = self.model.preprocess(src_img*255)
            src_img_feature = self.model.image_encoder(src_img_norm).reshape(1, -1)
            if tar_img != None:
                tar_img_norm = self.model.preprocess(tar_img*255)
                tar_img_feature = self.model.image_encoder(tar_img_norm).reshape(1, -1)
            src_adv = src_img.detach() + torch.empty_like(src_img).uniform_(-self.args.epsilon, self.args.epsilon)
            src_adv = torch.clamp(src_adv, 0.0, 1.0)
            momentum = torch.zeros_like(src_adv).detach()
            # momentum = torch.zeros((B,C,H*(2**(self.args.gamma-1)),W*(2**(self.args.gamma-1)))).type_as(src_img)
            stacked_kernel = self.stacked_kernel.type_as(src_img)
        for t in range(self.args.num_iter):
            src_adv.requires_grad_()
            with torch.set_grad_enabled(True):
                adv_grad = torch.zeros_like(src_adv).detach().type_as(src_adv).float()
                cost=0
                for s in range(self.args.scale):
                    src_adv_m = src_adv / (2 ** s)
                    if tar_img == None:
                        src_adv_norm = self.model.preprocess(src_adv_m*255)
                        if self.args.is_dim:
                            src_adv_norm=self.input_diversity(src_adv_norm)
                        loss = self.cal_cos_loss(self.rst(src_adv_norm,beta=self.args.beta), src_img_feature)
                    else:
                        src_adv_norm = self.model.preprocess(src_adv_m*255)
                        if self.args.is_dim:
                            src_adv_norm=self.input_diversity(src_adv_norm)
                        loss = self.cal_cos_loss(self.rst(src_adv_norm, beta=self.args.beta),
                                                 src_img_feature,tar_img_feature)
                    grad = torch.autograd.grad(loss, [src_adv], grad_outputs=torch.ones(loss.size()).cuda())[0]
                    adv_grad = adv_grad + grad.detach().float()
                    cost = cost + loss.detach().cpu().numpy().mean()
                adv_grad = adv_grad / self.args.scale
            if self.args.is_ti:
                adv_grad = F.conv2d(adv_grad, stacked_kernel, stride=1, padding=int((self.len_kernel - 1) / 2), groups=3)
            if self.args.momentum>0:
                grad_norm = torch.norm(nn.Flatten()(adv_grad), p=1, dim=1)  # 1,2,float('inf')
                adv_grad = adv_grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
                adv_grad = adv_grad + momentum.float() * self.args.momentum
                momentum = adv_grad
            src_adv.data = src_adv.detach() - self.args.alpha * adv_grad.detach().sign()
            src_adv = torch.min(torch.max(src_adv, src_img - self.args.epsilon), src_img + self.args.epsilon)
            src_adv = torch.clamp(src_adv, 0.0, 1.0).detach()
            if t % self.args.print_iter == 0 and self.args.print_iter != 0 and self.args.print_iter <= self.args.num_iter:
                print("t:",t,'cost:', cost)
        return src_adv



