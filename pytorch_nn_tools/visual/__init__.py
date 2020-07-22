import numpy as np
import torch
import torchvision.transforms as transforms

imagenet_stats = dict(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])


class UnNormalize_(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class ImageShow:
    def __init__(self, ax, num_classes, interp_img='nearest', interp_mask='nearest', to_pil_image=transforms.ToPILImage(),
                 cmap_mask='tab20',
                 unnormalize_img=UnNormalize_(**imagenet_stats)
                 ):
        self.ax = ax
        self.interp_img = interp_img
        self.interp_mask = interp_mask
        self.num_classes = num_classes
        self.to_pil_image = to_pil_image
        self.cmap_mask = cmap_mask
        self.unnormalize_img = unnormalize_img

    def show_mask(self, tensor, size=None, imshow_kwargs={}):
        img = self.to_pil_image(tensor.detach().type(torch.IntTensor))
        if size:
            img = img.resize(size)
        self.ax.imshow(np.asarray(img), cmap=self.cmap_mask,
                       interpolation=self.interp_mask, vmin=0, vmax=self.num_classes,
                       **imshow_kwargs)
        return self

    def show_image(self, tensor, size=None, imshow_kwargs={}):
        img = self.to_pil_image(self.unnormalize_img(tensor.detach().clone().cpu()))
        if size:
            img = img.resize(size)
        self.ax.imshow(np.asarray(img), interpolation=self.interp_img, **imshow_kwargs)
        return self


