import numpy as np
from .models import dist_model as dm

from PIL import Image
import torchvision.transforms.functional as TF

'''
example usage:
    lpips = LPIPS()
    lpips.reset_accumulation()
    for multimodal_fakes in tqdm(task_fakes):
        lpips.compute_and_accumulate(multimodal_fakes)
    task_lpips_mean = lpips.mean()

'''


def read_image(file, normalize=True, unsqueeze=True):
    image = Image.open(file).convert('RGB')
    x = TF.to_tensor(image)
    if normalize:
        x = (x - 0.5) * 2
    if unsqueeze:
        x = x.unsqueeze(0)
    return x


class LPIPS():
    def __init__(self):
        self.dm = dm.DistModel()
        self.dm.initialize(model='net-lin', net='alex', use_gpu=True, version='0.1')
        self.reset_accumulation()

    def __call__(self, images, return_matrix=False):
        '''
        compute pairwise distance between images across elements of iterable
        example:
            images=[torch.zeros((1, C, H, W)) for _ in 10]  # 10 elements with batch_size=1
            lpips(images) returns 1 x 10 x 10 symmetric matrix containing 10 x 9 / 2 pairwise distances

            images=[torch.zeros((B, C, H, W)) for _ in 10]  # 10 elements with batch_size=B
            lpips(images) returns B x 10 x 10 symmetric matrix containing 10 x 9 / 2 pairwise distances for each batch
        args:
            images = iterable of images as tensors, B x C x H x W
        returns:
            B-dim vector where B = batch_size = images[0].size()
            if return_matrix:
                B-by-N-by-N matrix where N = len(images)
        '''
        num_images = len(images)
        assert num_images >= 2, "LPIPS got less than 2 images: %d" % num_images
        assert len(images[0].shape) == 4, "LPIPS requires batched images: [B x C x H x W]"
        batch_size = images[0].size(0)

        if num_images == 2:
            return self.dm.forward(images[0], images[1])

        dists = np.zeros((batch_size, num_images, num_images))
        for i in range(num_images - 1):
            for j in range(i+1, num_images):
                dist = self.dm.forward(images[i], images[j])
                dists[:, i, j] = dist

        if not return_matrix:
            return dists.sum(axis=(1, 2)) / (num_images * (num_images - 1) / 2)
        else:
            idx_lower = np.tril_indices(num_images, -1)
            for bi in range(batch_size):
                dists[bi, idx_lower] = dists.T[bi, idx_lower]  # make the matrix symmetric
            return dists

    def reset_accumulation(self):
        self.dists = np.empty(0)

    def compute_and_accumulate(self, images):
        if isinstance(images[0], str):
            images = [read_image(fname) for fname in images]
        dist = self(images)  # __call__(images)
        self.dists = np.concatenate([self.dists, dist])

    def mean(self):
        return self.dists.mean()
