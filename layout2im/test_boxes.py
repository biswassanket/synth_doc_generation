#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:46:52 2021

@author: sanket
"""

import torch
import argparse
from models.generator import Generator
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from data.coco_custom_mask import get_dataloader as get_dataloader_coco
from data.publaynet_custom_mask import get_dataloader as get_dataloader_publaynet
from utils.data import imagenet_deprocess_batch
from scipy.misc import imsave
import os
from pathlib import Path
import torch.backends.cudnn as cudnn
import numpy as np
## NEW IMPORTS

from tqdm import tqdm
from torchvision.utils import save_image
from utils.miscs import draw_bbox_batch, split_boxes_to_img, collect_boxes_with_dups

# colors = [(0, 255, 0),(0,0,0),(255,0,0),(0,0,255),(128,128,128),(255,96,208),(255,224,32),(0,192,0),(0,32,255),(255,208,160), (224, 224, 224)]


# def str2bool(v):
#     return v.lower() == 'true'


# def draw_bbox_batch(images, bbox_sets, objs):
#     device = images.device
#     results = []
#     images = images.cpu().numpy()
#     images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)), dtype=np.float32)
#     for image, bbox_set in zip(images, bbox_sets):
#         for i, bbox in enumerate(bbox_set):
#             if all(bbox == 0):
#                 continue
#             else:
#                 try:
#                     image = draw_bbox(image, bbox, i, objs)
#                 except:
#                     continue
#         results.append(image)

#     images = np.stack(results, axis=0)
#     images = np.transpose(images, (0, 3, 1, 2))
#     images = torch.from_numpy(images).float().to(device)
#     return images


# def draw_bbox(image, bbox, i, objs):
#     im = Image.fromarray(np.uint8(image * 255))
#     draw = ImageDraw.Draw(im)

#     h, w, _ = image.shape
#     c1 = (round(float(bbox[0] * w)), round(float(bbox[1] * h)))
#     c2 = (round(float(bbox[2] * w)), round(float(bbox[3] * h)))

#     draw.rectangle([c1, c2], outline=colors[i])
#     draw.text((5, 5), "aa", font=ImageFont.truetype("arial"), fill=(255, 255, 0))

#     output = np.array(im)/255

#     return output


def main(config):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_save_dir = config.results_dir
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)

    if config.dataset == 'vg':
        train_data_loader, val_data_loader = get_dataloader_vg(batch_size=config.batch_size, VG_DIR=config.vg_dir)
    elif config.dataset == 'coco':
        train_data_loader, val_data_loader = get_dataloader_coco(batch_size=config.batch_size, COCO_DIR=config.coco_dir)
    elif config.dataset == 'publaynet':
        train_data_loader, val_data_loader = get_dataloader_publaynet(batch_size=config.batch_size, COCO_DIR=config.coco_dir)
    vocab_num = train_data_loader.dataset.num_objects

    assert config.clstm_layers > 0
    netG = Generator(num_embeddings=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim, clstm_layers=config.clstm_layers).to(device)

    print('load model from: {}'.format(config.saved_model))
    netG.load_state_dict(torch.load(config.saved_model))
    
    data_loader = val_data_loader
    data_iter = iter(data_loader)
    
    with torch.no_grad():
        netG.eval()
        
        for i, batch in enumerate(data_iter):
            print('batch {}'.format(i))
            imgs, objs, boxes, masks, obj_to_img = batch
            
            imgs, objs, boxes, masks, obj_to_img = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img
            
            # Generate fake images
            
            H, W = masks.shape[2], masks.shape[3]
            boxes_original = boxes
            objs_original = objs
            obj_to_img_original = obj_to_img
            
            for j in range(5):
                new_mask, new_boxes, new_objs, new_obj_to_img = [], [], [], []
                for im_idx, im in enumerate(imgs):
                    obj_idx = obj_to_img_original == im_idx
                    boxes_idx = boxes[obj_idx]
                    sampling_idxs = torch.randperm(boxes_idx.shape[0])[:torch.randint(1, boxes_idx.shape[0],(1,))]
                    new_boxes.append(boxes_idx[sampling_idxs])
                    new_obj_to_img.append(obj_to_img_original[obj_idx][sampling_idxs])
                    new_objs.append(objs[obj_idx][sampling_idxs])
                    new_mask.append(masks[obj_idx][sampling_idxs])

                new_boxes = torch.cat(new_boxes)
                new_obj_to_img = torch.cat(new_obj_to_img)
                new_objs = torch.cat(new_objs)
                new_mask = torch.cat(new_mask)
                
                z= torch.randn(new_objs.size(0), config.z_dim)
                z= z.to(device)
                
                output = netG(imgs, new_objs, new_boxes, new_mask, new_obj_to_img, z)
                
                crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec = output
                # Generate set of boxes (layouts)
                # boxes_set =[]
                # for img in range(imgs.shape[0]):
                #     idx = list(torch.nonzero(obj_to_img == img).view(-1).numpy())
                #     boxes_set.append(boxes[idx])
                
                # boxes_set= split_boxes_to_img(boxes, obj_to_img, config.batch_size)
                # img_input = imagenet_deprocess_batch(imgs, to_byte=False)
                # img_rec = imagenet_deprocess_batch(img_rand, to_byte=False)
                img_rand = imagenet_deprocess_batch(img_rand, to_byte=False)
                # img_rand_box = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
                # img_rand_box = draw_bbox_batch(img_rand, boxes_set)
                # img_rec_box = draw_bbox_batch(img_rec, boxes_set)
                # img_input_box = draw_bbox_batch(img_input, boxes_set)
                # Save generated images
                
                for k in range(img_rand.shape[0]):
                    img_np = img_rand[k].numpy().transpose(1,2,0)
                    img_path = os.path.join(result_save_dir, 'img{:06d}_{}.png'.format(i*config.batch_size+k, j))
                    imsave(img_path, img_np)
                
                # for j in range(imgs.shape[0]):
                #     img_np = img_input_box[j].numpy().transpose(1,2,0)
                #     img_path = os.path.join(result_save_dir, 'img{:06d}.png'.format(i*config.batch_size+j))
                #     imsave(img_path, img_np)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datasets configuration
    parser.add_argument('--dataset', type=str, default='publaynet')
    parser.add_argument('--vg_dir', type=str, default='datasets/vg')
    parser.add_argument('--coco_dir', type=str, default='/home/sanket/Documents/PubLayNet/')

    # Model configuration
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--object_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--resi_num', type=int, default=6)
    parser.add_argument('--clstm_layers', type=int, default=3)

    # Model setting
    # parser.add_argument('--saved_model', type=str, default='checkpoints/pretrained/netG_coco.pkl')
    parser.add_argument('--saved_model', type=str, default='/home/sanket/Documents/synth_doc_generation_old/checkpoints/layout2im_publaynet/models/iter-300000_netG.pkl')

    # test cases
    # parser.add_argument('--test_case', type=str, default='rand', choices=['rand', 'ref'])
    # parser.add_argument('--num_multimodal', type=int, default=2)
    config = parser.parse_args()
    # config.results_dir = 'checkpoints/pretrained_results_{}'.format(config.dataset)
    # config.results_dir = '/home/sanket/Documents/synth_doc_generation/checkpoints/layout2im_publaynet/results/pretrained_results_{}'.format(config.dataset)
    config.results_dir = '/home/sanket/Documents/synth_doc_generation_old/checkpoints/layout2im_publaynet/samples/new_results_input'
    print(config)

    main(config)
                
                
                
            
        
            
            
            
            
            
            
            
            
            
    
