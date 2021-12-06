# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/rwightman/pytorch-image-models)
# ------------------------------------------------------------------------
""" Random Erasing (Cutout)
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random

import torch


def _get_pixels(
    per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"
):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty(
            (patch_size[0], 1, 1), dtype=dtype, device=device
        ).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        syncronized=False,
        device="cpu",
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.syncronized = syncronized
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        if self.syncronized and len(img.size()) == 4:
            img = img.view(-1, img_h, img_w)
            chan = img.size(0)
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )

        for _ in range(count):
            for attempt in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    # C T H W
    def __call__(self, img):
        if len(img.size()) == 3:
            raise ValueError("Expected a 4D tensor.")
        else:
            img = img.permute(1, 0, 2, 3).contiguous()  # -> T, C, H, W
            frame, chan, img_h, img_w = img.size()
            if not self.syncronized:
                for i in range(frame):
                    self._erase(img[i], chan, img_h, img_w, img[i].dtype)
            else:
                self._erase(img, chan, img_h, img_w, img.dtype)
            img = img.permute(1, 0, 2, 3)  # -> C, T, H, W
        return img
