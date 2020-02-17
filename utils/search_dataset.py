import os
import torch
import numpy as np
import ipdb
from . import img_process 

def get_real_sketch_batch(batch_size, img_name_list, dataset_filter):
    img_name_list_all = np.array([x.strip() for x in open(img_name_list).readlines()])#268
    img_name_list     = []
    for idx, i in enumerate(img_name_list_all):
        for j in dataset_filter:
            if j in i:
                img_name_list.append(i)
                break
    sketch_name_list = [x.replace('train_photos', 'train_sketches') for x in img_name_list] 
    sketch_name_list = np.array(sketch_name_list)
    img_batch = np.random.choice(sketch_name_list, batch_size, replace=False)
    img_batch = [img_process.read_img_var(x, 0, size=(224, 224)) for x in img_batch]#len=batch_size, img_batch[0]:[1,1,224,224]
    return torch.stack(img_batch).squeeze(1)


def find_photo_sketch_batch(photo_batch, dataset_path, img_name_list, vgg_model, 
        topk=1, dataset_filter=['CUHK_student', 'AR'], compare_layer=['r51']):
    """
    Search the dataset to find the topk matching image.
    """
    # photo_batch [4,3,224,224], dataset_path: ./data/cufs_feature_dataset.pth, 
    # img_name_list: ./data/cufs_reference_img_list.txt
    # topk=1
    dataset_all       = torch.load(dataset_path) #[1004,512,14,14]
    dataset_all       = torch.autograd.Variable(dataset_all.type_as(photo_batch.data))
    img_name_list_all = np.array([x.strip() for x in open(img_name_list).readlines()]) #268
    img_name_list     = []
    dataset_idx       = []
    for idx, i in enumerate(img_name_list_all):
        for j in dataset_filter:
            if j in i:
                img_name_list.append(i)
                dataset_idx.append(idx)
                break
    # choose img in th dataset_filter, img_name_list: 268
    dataset = dataset_all[dataset_idx] #[268,512,14,14]
    img_name_list = np.array(img_name_list)# 268

    photo_feat = vgg_model(img_process.subtract_mean_batch(photo_batch), compare_layer)[0]  #[4,512,14,14]
    photo_feat = torch.nn.functional.normalize(photo_feat, p=2, dim=1).view(photo_feat.size(0), photo_feat.size(1), -1) #[4,512,196]
    dataset    = torch.nn.functional.normalize(dataset, p=2, dim=1).view(dataset.size(0), dataset.size(1), -1) #[268,512,196]
    img_idx    = []
    for i in range(photo_feat.size(0)): # for every image
        dist = photo_feat[i].unsqueeze(0) * dataset #[268,512,196]
        dist = torch.sum(dist, -1) #[268,512]
        dist = torch.sum(dist, -1) #[268]
        _, best_idx = torch.topk(dist, topk, 0) # choose the top k=5 nearest images for photo[i]
        img_idx += best_idx.data.cpu().tolist()

    match_img_list    = img_name_list[img_idx] # this is for photo, 20
    match_sketch_list = [x.replace('train_photos', 'train_sketches') for x in match_img_list] # this is for sketch

    match_img_batch    = [img_process.read_img_var(x, size=(224, 224)) for x in match_img_list] # len:20, match_img_batch[0]=[1,3,2224,224]
    match_sketch_batch = [img_process.read_img_var(x, size=(224, 224)) for x in match_sketch_list] # len:20, match_sketch_batch[0]=[1,3,224,224]
    match_sketch_batch, match_img_batch = torch.stack(match_sketch_batch).squeeze(), torch.stack(match_img_batch).squeeze() #[20,3,224,224]

    return match_sketch_batch, match_img_batch

def select_random_batch(ref_img_list, batch_size, dataset_filter=['CUHK_student', 'AR']):
    ref_img_list_all = np.array([x.strip() for x in open(ref_img_list).readlines()])
    ref_img_list     = []
    for idx, i in enumerate(ref_img_list_all):
        for j in dataset_filter:
            if j in i:
                ref_img_list.append(i)
                break
    ref_img_list = np.array(ref_img_list)

    selected_ref_img = np.random.choice(ref_img_list, batch_size, replace=False)
    selected_ref_sketch = [x.replace('train_photos', 'train_sketches') for x in selected_ref_img]

    selected_ref_batch    = [img_process.read_img_var(x, size=(224, 224)) for x in selected_ref_img]
    selected_sketch_batch = [img_process.read_img_var(x, size=(224, 224)) for x in selected_ref_sketch]
    selected_sketch_batch, selected_ref_batch = torch.stack(selected_sketch_batch).squeeze(1), torch.stack(selected_ref_batch).squeeze(1)
    return selected_ref_batch, selected_sketch_batch

