import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_bbox_batch(images, bbox_sets):
    device = images.device
    results = []
    images = images.cpu().numpy()
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)), dtype=np.float32)
    for image, bbox_set in zip(images, bbox_sets):
        for bbox in bbox_set:
            if all(bbox == 0):
                continue
            else:
                image = draw_bbox(image, bbox)

        results.append(image)

    images = np.stack(results, axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).float().to(device)
    return images


def draw_bbox(image, bbox):
    im = Image.fromarray(np.uint8(image * 255))
    draw = ImageDraw.Draw(im)

    h, w, _ = image.shape
    c1 = (round(float(bbox[0] * w)), round(float(bbox[1] * h)))
    c2 = (round(float(bbox[2] * w)), round(float(bbox[3] * h)))

    draw.rectangle([c1, c2], outline=(0, 255, 0))

    output = np.array(im)/255

    return output

def str2bool(v):
    return v.lower() == 'true'



def split_boxes_to_img(boxes, obj_to_img, batch_size):
    batch_size = len(boxes)
    boxes_split = []
    for i in range(batch_size):
        flag = obj_to_img == i
        bb = boxes[flag]
        boxes_split.append(bb)
    return boxes_split


def elapsed(t_start):
    now = time.time()
    s = now - t_start
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)


def collect_boxes_with_dups(objs, boxes, obj_to_img, least_dups=2):
    num_objs = len(objs)
    temp = objs.expand((num_objs, num_objs))
    comparison = temp == temp.t()
    comparison.sum(dim=1)
    flags = comparison.sum(dim=1) >= least_dups
    objs_with_duplicates = objs[flags]
    boxes_with_duplicates = boxes[flags]
    obj_to_img_with_duplicates = obj_to_img[flags]
    return objs_with_duplicates, boxes_with_duplicates, obj_to_img_with_duplicates


def shuffle_obj_to_img(objs, obj_to_img, least_dups=3):
    num_objs = len(objs)
    num_alts = least_dups - 1
    expanded = objs.expand(num_objs, num_objs)
    comparison = expanded == expanded.t()
    comparison = comparison - torch.eye(num_objs).byte().to(comparison.device)

    # O -> 2 alts = O x 2
    obj_to_img_alts_idx = torch.zeros(num_objs, num_alts).long()
    alts_idx_list = []
    for oi in range(num_objs):
        alts_idx = comparison[oi].nonzero().squeeze(dim=1)
        flag_other_image = obj_to_img[oi] != obj_to_img[alts_idx]
        alts_idx = alts_idx[flag_other_image]
        alts_idx_list.append(alts_idx)
        if len(alts_idx) >= num_alts:
            obj_to_img_alts_idx[oi] = alts_idx[:num_alts]
        else:
            obj_to_img_alts_idx[oi] = obj_to_img[oi:oi+1].repeat(num_alts)
    obj_to_img_alts = obj_to_img[obj_to_img_alts_idx].t()
    return obj_to_img_alts