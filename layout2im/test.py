import torch
import argparse
from models.generator import Generator
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from data.coco_custom_mask import get_dataloader as get_dataloader_coco
from data.publaynet_custom_mask import get_dataloader as get_dataloader_publaynet
from utils.data import imagenet_deprocess_batch
from imageio import imwrite
import os
from pathlib import Path
import torch.backends.cudnn as cudnn

## NEW IMPORTS

from tqdm import tqdm
from torchvision.utils import save_image
from utils.miscs import draw_bbox_batch, split_boxes_to_img, collect_boxes_with_dups
# from metrics.lpips.lpips import LPIPS


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

    # _lpips = LPIPS()

    data_loader = val_data_loader
    data_iter = iter(data_loader)
    with torch.no_grad():
        netG.eval()
        for i, batch in tqdm(enumerate(data_iter)):
            print('batch {}'.format(i))
            imgs, objs, boxes, masks, obj_to_img = batch
            z = torch.randn(objs.size(0), config.z_dim)
            imgs, objs, boxes, masks, obj_to_img, z = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img, z.to(device)
            
            fake_group = []


            if config.test_case == 'rand':
                boxes_split = split_boxes_to_img(boxes, obj_to_img, config.batch_size)
                imgs_denorm = imagenet_deprocess_batch(imgs, to_byte=False)
                imgs_with_box = draw_bbox_batch(imgs_denorm, boxes_split)
                visual = [imgs_with_box]
                for zi in range(config.num_multimodal):
                    z = torch.randn(objs.size(0), config.z_dim).to(device)

                    # Generate fake image
                    output = netG(imgs, objs, boxes, masks, obj_to_img, z)
                    crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec = output
                    img_rand = imagenet_deprocess_batch(img_rand, to_byte=False)
                    img_rec = imagenet_deprocess_batch(img_rec, to_byte=False)
            # boxes_split = split_boxes_to_img(boxes, obj_to_img, config.batch_size)
            # img_rec = imagenet_deprocess_batch(img_rec, to_byte=False)
            # imgs_with_box = draw_bbox_batch(img_rec,boxes_split)
            # visual = [imgs_with_box]
            # visual = torch.cat(visual, dim=3)
            # img_path = os.path.join(result_save_dir, 'img{:06d}.png'.format(i*config.batch_size))
            # save_image(visual,img_path)
                    if zi == 0:
                        visual.append(img_rec)
                    visual.append(img_rand)
                    fake_group.append(img_rand)

            elif config.test_case == 'ref':
                imgs_denorm = imagenet_deprocess_batch(imgs, to_byte=False)
                # draw boxes: basic
                boxes_split = split_boxes_to_img(boxes, obj_to_img, config.batch_size)
                imgs_with_box = draw_bbox_batch(imgs_denorm, boxes_split)
                visual = [imgs_with_box]

                # draw boxes with alternatives only
                objs_with_dups, boxes_with_dups, obj_to_img_with_dups = collect_boxes_with_dups(objs, boxes, obj_to_img, least_dups=config.num_multimodal)

                boxes_split_with_dups = split_boxes_to_img(boxes_with_dups, obj_to_img_with_dups, config.batch_size)
                if len(boxes_split_with_dups) == 0:
                    print('not enough alternative objects in batch to be used as references: ', i, 'th batch')
                    img_path = os.path.join(result_save_dir, 'img{:06d}_{}.png'.format(i, config.test_case))
                    visual = torch.cat(visual, dim=3)
                    save_image(visual, img_path, nrow=1)
                    print('saved to ', img_path)
                    continue
                imgs_with_box_dups = draw_bbox_batch(imgs_denorm, boxes_split_with_dups)
                visual.append(imgs_with_box_dups)

                # img_ref_multimodal = netG.ref_guide(imgs, objs, boxes,  masks, obj_to_img, config.num_multimodal)
                img_ref_multimodal = netG(imgs, objs, boxes,  masks, obj_to_img, config.num_multimodal)
                for img_ref in img_ref_multimodal:
                    img_ref = imagenet_deprocess_batch(img_ref, to_byte=False)
                    visual.append(img_ref)
                    fake_group.append(img_ref)
            else:
                raise ValueError

            # _lpips.compute_and_accumulate(fake_group)


            if i < 5:
                img_path = os.path.join(result_save_dir, 'img{:06d}_{}.png'.format(i, config.test_case))
                visual = torch.cat(visual, dim=3)
                save_image(visual, img_path, nrow=1)

        # print('lpips: ', _lpips.mean())


            # # Save the generated images
            # for j in range(img_rand.shape[0]):
            #     img_np = img_rand[j].numpy().transpose(1, 2, 0)
            #     img_path = os.path.join(result_save_dir, 'img{:06d}.png'.format(i*config.batch_size+j))
            #     imwrite(img_path, img_np)


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
    parser.add_argument('--saved_model', type=str, default='/home/sanket/Documents/synth_doc_generation/checkpoints/layout2im_publaynet/models/iter-300000_netG.pkl')

    # test cases
    parser.add_argument('--test_case', type=str, default='rand', choices=['rand', 'ref'])
    parser.add_argument('--num_multimodal', type=int, default=2)
    config = parser.parse_args()
    # config.results_dir = 'checkpoints/pretrained_results_{}'.format(config.dataset)
    # config.results_dir = '/home/sanket/Documents/synth_doc_generation/checkpoints/layout2im_publaynet/results/pretrained_results_{}'.format(config.dataset)
    config.results_dir = '/home/sanket/Documents/synth_doc_generation/checkpoints/layout2im_publaynet/samples'
    print(config)

    main(config)
