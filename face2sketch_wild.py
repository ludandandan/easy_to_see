from __future__ import print_function
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR 
from torchvision import transforms

import argparse
import os
import numpy as np
from time import time
from datetime import datetime
import itertools
import copy
from glob import glob

from utils.face_sketch_data import * 
from models.networks import SketchNet, DNet
from models.vgg19 import vgg19 
from utils import loss 
from utils import img_process 
from utils import search_dataset
from utils import logger
from utils import utils
from utils.metric import avg_score
import ipdb

def cmd_option():
    arg_parser = argparse.ArgumentParser(description='CMD arguments for the face sketch network')
    arg_parser.add_argument('train_eval', type=str, default='train', help='Train or eval')
    arg_parser.add_argument('--gpus', type=str, default='0', help='Which gpus to train the model')
    arg_parser.add_argument('--train-data', type=str, nargs='*', 
            default=["./data/AR/train_photos", "./data/CUHK_student/train_photos", "./data/XM2VTS/train_photos", "./data/CUFSF/train_photos"], help="Train data dir root")
    arg_parser.add_argument('--resume', type=int, default=0, help='Resume training or not')
    arg_parser.add_argument('--train-style', type=str, default='cufs', help='Styles used to train')
    arg_parser.add_argument('--seed', type=int, default=1234, help='Random seed for training')
    arg_parser.add_argument('--batch-size', type=int, default=6, help='Train batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    arg_parser.add_argument('--epochs', type=int, default=40, help='Training epochs to generate')
    arg_parser.add_argument('--weight-root', type=str, default='./weight', help='Weight saving path')
    arg_parser.add_argument('--vgg19-weight', type=str, default='/home/cfchen/pytorch_models/vgg_conv.pth',
                                                        help='Pretrained vgg19 weight path')
    arg_parser.add_argument('--Gnorm', type=str, default='IN', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--Dnorm', type=str, default='None', help="Instance(IN) normalization or batch(BN) normalization")
    arg_parser.add_argument('--flayers', type=int, nargs=5, default=[0, 0, 1, 1, 1], help="Layers used to calculate feature loss")
    arg_parser.add_argument('--weight', type=float, nargs=3, default=[1e0, 1e3, 1e-5], help="MSE loss weight, Feature loss weight, and total variation weight")
    arg_parser.add_argument('--topk', type=int, default=1, help="Topk image choose to match input photo")
    arg_parser.add_argument('--meanshift', type=int, default=20, help="mean shift of the predicted sketch.")
    arg_parser.add_argument('--other', type=str, default='', help="Other information")
    
    arg_parser.add_argument('--test-dir', type=str, default='', help='Test image directory')
    arg_parser.add_argument('--test-gt-dir', type=str, default='', help='Test ground truth image directory')
    arg_parser.add_argument('--result-dir', type=str, default='./result', help='Result saving directory')
    arg_parser.add_argument('--test-weight-path', type=str, default='', help='Test model path')
    return arg_parser.parse_args()

def train(args):
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -------------------- Load data ----------------------------------
    transform = transforms.Compose([
                    Rescale((224, 224)), 
                    ColorJitter(0.5, 0.5, 0.5, 0.3, 0.5),
                    ToTensor(),
        ])
    dataset = FaceDataset(args.train_data, True, transform=transform) #utils/face_sketch_data.py
    data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=4) 

    # load average image and make weight
    weight_img = img_process.read_sketch_var('./data/CUFS/ravg.jpg', size=(224,224))
    # [4,3,224,224]
    weight_img = weight_img.expand([4,3,224,224])

    # ----------------- Define networks ---------------------------------
    Gnet= SketchNet(in_channels=3, out_channels=1, norm_type=args.Gnorm)
    Dnet = DNet(norm_type=args.Dnorm)
    Dnet56 = DNet(norm_type=args.Dnorm)
    Dnet112 = DNet(norm_type=args.Dnorm)
    vgg19_model = vgg19(args.vgg19_weight) 

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if len(gpu_ids) > 0:
        Gnet.cuda()
        Dnet.cuda()
        Gnet = nn.DataParallel(Gnet, device_ids=gpu_ids) 
        Dnet = nn.DataParallel(Dnet, device_ids=gpu_ids)
        Dnet56 = nn.DataParallel(Dnet56, device_ids=gpu_ids)
        Dnet112 = nn.DataParallel(Dnet112, device_ids=gpu_ids)
        vgg19_model = nn.DataParallel(vgg19_model, device_ids=gpu_ids)

    Gnet.train()
    Dnet.train()
    Dnet56.train()
    Dnet112.train()

    if args.resume: #
        weights = glob(os.path.join(args.save_weight_path, '*-*.pth'))
        weight_path = sorted(weights)[-1][:-5]
        Gnet.load_state_dict(torch.load(weight_path + 'G.pth'))
        Dnet.load_state_dict(torch.load(weight_path + 'D.pth'))
        Dnet56.load_state_dict(torch.load(weight_path + 'D56.pth'))
        Dnet112.load_state_dict(torch.load(weight_path + 'D112.pth'))

    # ---------------- set optimizer and learning rate ---------------------
    args.epochs = np.ceil(args.epochs * 1000 / len(dataset))
    args.epochs = max(int(args.epochs), 4)
    ms = [int(1./4 * args.epochs), int(2./4 * args.epochs)]

    optim_G = Adam(Gnet.parameters(), args.lr)
    optim_D = Adam(Dnet.parameters(), args.lr)
    optim_D56 = Adam(Dnet56.parameters(), args.lr)
    optim_D112 = Adam(Dnet112.parameters(), args.lr)
    scheduler_G = MultiStepLR(optim_G, milestones=ms, gamma=0.1) 
    scheduler_D = MultiStepLR(optim_D, milestones=ms, gamma=0.1)
    scheduler_D56 = MultiStepLR(optim_D56, milestones=ms, gamma=0.1)
    scheduler_D112 = MultiStepLR(optim_D112, milestones=ms, gamma=0.1)
    mse_crit  = nn.MSELoss()
    l1_crit = nn.L1Loss()
    
    # ---------------------- Define reference styles and feature loss layers ----------        
    if args.train_style == 'cufs':
        ref_style_dataset = ['CUHK_student', 'AR', 'XM2VTS']
        ref_feature       = './data/cufs_feature_dataset.pth'
        ref_img_list      = './data/cufs_reference_img_list.txt'
    elif args.train_style == 'cufsf':
        ref_style_dataset = ['CUFSF']
        ref_feature       = './data/cufsf_feature_dataset.pth'
        ref_img_list      = './data/cufsf_reference_img_list.txt'
    else:
        assert 1==0, 'Train style {} not supported.'.format(args.train_style)

    vgg_feature_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    feature_loss_layers = list(itertools.compress(vgg_feature_layers, args.flayers)) 

    log = logger.Logger(args.save_weight_path)

    for e in range(args.epochs):
        scheduler_G.step()
        scheduler_D.step()
        sample_count = 0 
        for batch_idx, batch_data in enumerate(data_loader):
            # ---------------- Load data -------------------
            start = time()
            train_img, train_img_org, sketch, sketch56, sketch112 = [utils.tensorToVar(x) for x in batch_data]#[4,3,224,224], [4,1,224,224], [4,1,56,56], [4,1,112,112]
            topk_sketch_img, topk_photo_img = search_dataset.find_photo_sketch_batch( 
                            train_img_org, ref_feature, ref_img_list,
                            vgg19_model, dataset_filter=ref_style_dataset, topk=args.topk)
                            #topk_sketch_img[20,3,224,224],topk_photo_img[20,3,224,224]
            random_real_sketch = search_dataset.get_real_sketch_batch(train_img.size(0), ref_img_list, dataset_filter=ref_style_dataset) # random choose a batch of sketch image from training sketches feed to the D net [4,1,224,224]
            end           = time()
            data_time     = end - start
            sample_count += train_img.size(0) # train_img.size()=[4,3,224,224], sample count +4

            # ---------------- Model forward -------------------
            start = time()
            #fake_sketch = Gnet(train_img)  # [4,1,224,224]        
            fake_sketch56, fake_sketch112, fake_sketch = Gnet(train_img) # [4,1,56,56], [4,1,112,112],[4,1,224,224]
            fake_score = Dnet(fake_sketch) # [4,1,14,14]
            real_score = Dnet(random_real_sketch) # [4,1,14,14]
            
            fake56_score = Dnet56(fake_sketch56) #[4,1,4,4]
            real56_score = Dnet56(sketch56) #[4,1,7,7]
            fake112_score = Dnet112(fake_sketch112)
            real112_score = Dnet112(sketch112)

            real_label = torch.ones_like(fake_score) # [4,1,14,14]
            fake_label = torch.zeros_like(fake_score) # [4,1,14,14]

            real56_label = torch.ones_like(fake56_score)
            fake56_label = torch.zeros_like(fake56_score)
            real112_label = torch.ones_like(fake112_score)
            fake112_label = torch.zeros_like(fake112_score)


            # ----------------- Calculate loss and backward ------------------- 
            train_img_org_vgg   = img_process.subtract_mean_batch(train_img_org, 'face') # [4,3,224,224]
            topk_sketch_img_vgg = img_process.subtract_mean_batch(topk_sketch_img, 'sketch') #[20,3,224,224]
            topk_photo_img_vgg  = img_process.subtract_mean_batch(topk_photo_img, 'face') # [20,3,224,224]
            fake_sketch_vgg = img_process.subtract_mean_batch(fake_sketch.expand_as(train_img_org), 'sketch', args.meanshift) # meanshift=20, fake_sketch_vgg.shape[4,3,224,224]
            weight_img_vgg = img_process.subtract_mean_batch(weight_img, 'sketch') # ld add
            style_loss = loss.feature_mrf_loss_func(
                                fake_sketch_vgg, topk_sketch_img_vgg, weight_img_vgg, vgg19_model,#ld add
                                feature_loss_layers, [train_img_org_vgg, topk_photo_img_vgg], topk=args.topk) 
            tv_loss = loss.total_variation(fake_sketch)
            #l1_loss56 = l1_crit(fake_sketch56, sketch56)
            l1_loss56 = mse_crit(fake_sketch56, sketch56)
            #l1_loss112 = l1_crit(fake_sketch112, sketch112)
            l1_loss112 = mse_crit(fake_sketch112, sketch112)
            #l1_loss224 = l1_crit(fake_sketch, sketch)
            l1_loss224 = mse_crit(fake_sketch, sketch)

            # GAN Loss
            adv_loss = mse_crit(fake_score, real_label) * args.weight[1]
            adv_loss56 = mse_crit(fake56_score, real56_label) * args.weight[1]
            adv_loss112 = mse_crit(fake112_score, real112_label) * args.weight[1]
            tv_loss  = tv_loss * args.weight[2]
            l1_loss56 = l1_loss56*10
            l1_loss112 = l1_loss112*10
            l1_loss224 = l1_loss224*10
            
            #loss_G = style_loss * args.weight[0] + adv_loss + tv_loss 
            loss_G = style_loss*args.weight[0] + adv_loss + adv_loss56 + adv_loss112 + tv_loss + l1_loss56 + l1_loss112 + l1_loss224
            loss_D = 0.5 * mse_crit(fake_score, fake_label) + 0.5 * mse_crit(real_score, real_label) 
            loss_D56 = 0.5*mse_crit(fake56_score, fake56_label) + 0.5*mse_crit(real56_score, real56_label)
            loss_D112 = 0.5*mse_crit(fake112_score, fake112_label) + 0.5*mse_crit(real112_score, real112_label)

            # Update parameters 
            optim_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optim_D.step()
            optim_D56.zero_grad()
            loss_D56.backward(retain_graph=True)
            optim_D56.step()
            optim_D112.zero_grad()
            loss_D112.backward(retain_graph=True)
            optim_D112.step()

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            end = time()
            train_time = end - start

            # ----------------- Print result and log the output ------------------- 
            log.iterLogUpdate(loss_G.data[0])
            if batch_idx % 100 == 0:
                log.draw_loss_curve()

            msg = "{:%Y-%m-%d %H:%M:%S}\tEpoch [{:03d}/{:03d}]\tBatch [{:03d}/{:03d}]\tData: {:.2f}  Train: {:.2f}\tLoss: G-{:.4f}, Adv-{:.4f}, tv-{:.4f}, D-{:.4f}, L1_56-{:.4f}, L1_112-{:.4f}, L1_224-{:.4f}".format(
                            datetime.now(), 
                            e, args.epochs, sample_count, len(dataset),
                            data_time, train_time, *[x.data[0] for x in [loss_G, adv_loss, tv_loss, loss_D, l1_loss56, l1_loss112, l1_loss224]])
            print(msg)
            log_file = open(os.path.join(args.save_weight_path, 'log.txt'), 'a+')
            log_file.write(msg + '\n')
            log_file.close()
        
        save_weight_name = "epochs-{:03d}-".format(e)
        G_cpu_model = copy.deepcopy(Gnet).cpu() 
        D_cpu_model = copy.deepcopy(Dnet).cpu()
        D56_cpu_model = copy.deepcopy(Dnet56).cpu()
        D112_cpu_model = copy.deepcopy(Dnet112).cpu()
        torch.save(G_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'G.pth'))
        torch.save(D_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'D.pth'))
        torch.save(D56_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'D56.pth'))
        torch.save(D112_cpu_model.state_dict(), os.path.join(args.save_weight_path, save_weight_name+'D112.pth'))


def test(args):
    """
    Test image of a given directory. Calculate the quantitative result if ground truth dir is provided.
    """
    Gnet= SketchNet(in_channels=3, out_channels=1, norm_type=args.Gnorm)
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if len(gpu_ids) > 0:
        Gnet.cuda()
        Gnet = nn.DataParallel(Gnet, device_ids=gpu_ids) 
    Gnet.eval()
    Gnet.load_state_dict(torch.load(args.test_weight_path))

    utils.mkdirs(args.result_dir)    
    for img_name in os.listdir(args.test_dir):
        test_img_path = os.path.join(args.test_dir, img_name)
        test_img = img_process.read_img_var(test_img_path, size=(256, 256))
        _, _, face_pred= Gnet(test_img)

        sketch_save_path = os.path.join(args.result_dir, img_name)
        img_process.save_var_img(face_pred, sketch_save_path, (250, 200))
        print('Save sketch in', sketch_save_path)

    if args.test_gt_dir != 'none':
        print('------------ Calculating average SSIM (This may take for a while)-----------')
        avg_ssim = avg_score(args.result_dir, args.test_gt_dir, metric_name='ssim', smooth=False, verbose=True) 
        print('------------ Calculating smoothed average SSIM (This may take for a while)-----------')
        avg_ssim_smoothed = avg_score(args.result_dir, args.test_gt_dir, metric_name='ssim', smooth=True, verbose=True) 
        #print('------------ Calculating average FSIM (This may take for a while)-----------')
        #avg_fsim = avg_score(args.result_dir, args.test_gt_dir, metric_name='fsim', smooth=False, verbose=True) 
        #print('------------ Calculating smoothed average FSIM (This may take for a while)-----------')
        #avg_fsim_smoothed = avg_score(args.result_dir, args.test_gt_dir, metric_name='fsim', smooth=True, verbose=True) 
        print('Average SSIM: {}'.format(avg_ssim))
        print('Average SSIM (Smoothed): {}'.format(avg_ssim_smoothed))
        #print('Average FSIM: {}'.format(avg_fsim))
        #print('Average FSIM (Smoothed): {}'.format(avg_fsim_smoothed))

if __name__ == '__main__':
    args = cmd_option()
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    #torch.cuda.set_device(0)

    args.save_weight_dir = 'face2sketch-norm_G{}_D{}-top{}-style_{}-flayers{}-weight-{:.1e}-{:.1e}-{:.1e}-epoch{:02d}-{}'.format(
                        args.Gnorm, args.Dnorm, args.topk, args.train_style, "".join(map(str, args.flayers)),
                        args.weight[0], args.weight[1], args.weight[2], 
                        args.epochs, args.other) 
    args.save_weight_path = os.path.join(args.weight_root, args.save_weight_dir)

    if args.train_eval == 'train':
        print('Saving weight path', args.save_weight_path)
        utils.mkdirs(args.save_weight_path)
        train(args)
    elif args.train_eval == 'eval':
        test(args)


