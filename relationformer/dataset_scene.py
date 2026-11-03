"""
File that involves dataloaders for the Visual Genome dataset.
"""
_POS_FLAG_ANNOUNCED = False
import copy
import json
import os
import random
import h5py
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import torch
import torch.utils.data
import torchvision.transforms.functional as F
import transforms as T
from box_ops_2D import box_iou
from open_image import OIDataset


class visual_genome_loader(Dataset):
    def __init__(self, data_dicts, transforms, config, is_train=False,do_bd=False):
        self.data_dicts = data_dicts
        self.transforms = transforms
        self.BOX_SCALE = config.DATA.BOX_SCALE
        self.BG_EDGE_PER_IMG = config.DATA.BG_EDGE_PER_IMG
        self.FG_EDGE_PER_IMG = config.DATA.FG_EDGE_PER_IMG
        self.num_classes = config.MODEL.NUM_OBJ_CLS+1
        self.num_predicates = config.MODEL.NUM_REL_CLS+1
        self.is_train=is_train
        self.dataset = config.DATA.DATASET
        self.n_queries = config.MODEL.DECODER.NUM_QUERIES
        self.do_bd= do_bd

    def __getitem__(self, index):

        ##################### Load the image data
        image_unpadded = Image.open(self.data_dicts[index]['image']).convert('RGB')
        w, h = image_unpadded.size

        ##################### Load the box location data
        gt_boxes = self.data_dicts[index]['boxes'].copy()
        gt_classes = self.data_dicts[index]['label'].copy()
        ##################### Load the relationship data
        gt_rels = self.data_dicts[index]['edges'].copy()
        if len(gt_boxes)> self.n_queries-2:
            if self.is_train:
                idx = np.random.choice(len(gt_boxes), self.n_queries - 2, replace=False)
            else:
                # Keep eval deterministic when GT objects exceed query cap.
                idx = np.arange(self.n_queries - 2)
            idx_list = idx.tolist()
            idx_set = set(idx_list)
            idx_map = {old_i: new_i for new_i, old_i in enumerate(idx_list)}
            gt_boxes = gt_boxes[idx,:]
            gt_classes = gt_classes[idx]
            gt_rels = [[idx_map[rel[0]], idx_map[rel[1]], rel[2]] for rel in gt_rels if rel[0] in idx_set and rel[1] in idx_set]
            gt_rels = np.asarray(gt_rels, dtype=np.int64).reshape(-1, 3)
        # Boxes are already at BOX_SCALE
        if self.dataset=='VG':
            gt_boxes = gt_boxes/self.BOX_SCALE

            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes = gt_boxes.clip(0, 1)

                # # crop the image for data augmentation
                # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, self.BOX_SCALE, round_boxes=True)
            scale_ = np.maximum(h,w)
            # scale it back to image dimension
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]]*scale_
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]]*scale_
        #crop boxes that are too large
        if self.is_train:
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(None,h)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(None,w)



        # unique_comb, fwd_rels, inv_rels = get_obj_comb_and_rels(gt_rels, self.gt_classes[index].copy(), gt_boxes, self.use_bg_rels, self.require_overlap, self.FG_EDGE_PER_IMG, self.BG_EDGE_PER_IMG)
        # obj_rel_mat = get_obj_rels_mat(gt_rels, self.gt_classes[index])
        # gt_norm_boxes = get_normalized_boxes(gt_boxes.copy(), image_unpadded, self.BOX_SCALE)   #todo change as per im scale

        target = {}
        target["iscrowd"] = torch.zeros(gt_boxes.shape[0])
        target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float)
        target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        target["edges"] = torch.as_tensor(gt_rels, dtype=torch.int64)
        target["image_id"] = self.data_dicts[index]['id']
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            if self.do_bd:
                state_torch = torch.get_rng_state()
                state_py = random.getstate()
                state_np = np.random.get_state()

                cln_target_in = copy.deepcopy(target)
                cln_img, cln_target = self.transforms[0](image_unpadded, cln_target_in)

                torch.set_rng_state(state_torch)
                random.setstate(state_py)
                np.random.set_state(state_np)

                bd_target_in = copy.deepcopy(target)
                bd_img, bd_target = self.transforms[1](image_unpadded, bd_target_in)

                return cln_img, cln_target, bd_img, bd_target
            img, target = self.transforms(image_unpadded, target)

        return img, target

    def __len__(self):
        return len(self.data_dicts)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_obj_comb_and_rels(rels, gt_classes, gt_boxes, use_bg_rels, require_overlap, FG_EDGE_PER_IMG, BG_EDGE_PER_IMG):
    '''
    It create all posiible combination of rels which have some intersction or they have rels in gt
    :param rels:
    :param gt_classes:
    :param gt_boxes:
    :param use_bg_rels:
    :return:
    '''
    #create mix of bg and fg rels in sorted order unique does that
    unique_fg_edge = np.unique(np.sort(rels[:, :2], axis=1),axis=0)
    if use_bg_rels:
        all_comb =  np.arange(len(gt_classes))[:, None] != np.arange(len(gt_classes))[None]
        if require_overlap:
            all_comb = np.stack(np.nonzero(np.triu(all_comb & (box_iou_union_2d(gt_boxes, gt_boxes) > 0))), axis=-1)
            #all_comb = np.concatenate((all_comb,rels[:, 0:2]))
            #all_comb = np.concatenate((all_comb, unique_fg_edge))
            #unique_bg_edge = np.unique(all_comb, axis=0)
            #all_unique_rel = unique_bg_edge
        else:
            all_comb = np.stack(np.nonzero(np.triu(all_comb)), axis=-1)
    else:
        all_comb = unique_fg_edge
        assert 'This condition is not yet implemented'

    #now filter out the fg edge
    unique_bg_edge = []
    for edge in all_comb:
        if not (edge == unique_fg_edge).all(axis=1).any():
            unique_bg_edge.append(edge)
    #check if the number of rels exceeds the limit, then sample
    if len(unique_fg_edge)>FG_EDGE_PER_IMG:
        unique_fg_edge = np.asarray(random.sample(list(unique_fg_edge),k=FG_EDGE_PER_IMG))
    if len(unique_bg_edge)>BG_EDGE_PER_IMG and use_bg_rels:
        unique_bg_edge = np.asarray(random.sample(list(unique_bg_edge),k=BG_EDGE_PER_IMG))

    if len(unique_bg_edge)>0:
        all_comb = np.unique(np.concatenate((np.asarray(unique_bg_edge), unique_fg_edge)), axis=0)
    else:
        all_comb = unique_fg_edge

    all_comb = np.append(all_comb, np.zeros(all_comb.shape[0], dtype=np.int64)[:, None], axis=1)

    return get_fwd_inv_rels(all_comb, rels)

# def get_obj_rels_mat(gt_rels, gt_classes):
#     obj_rel_mat = np.full((len(gt_classes), len(gt_classes)), 0)
#     for rel in gt_rels:
#         obj_rel_mat[rel[0],rel[1]] = 1
#         obj_rel_mat[rel[1], rel[0]] = 1
#     return  obj_rel_mat


# def get_normalized_boxes(norm_boxes, image_unpadded, BOX_SCALE):
#     unscaled_img = np.array([BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0],
#                              BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1]])
#     norm_boxes = norm_boxes.astype(float)
#     norm_boxes[:, 0] /= unscaled_img[0]
#     norm_boxes[:, 1] /= unscaled_img[1]
#     norm_boxes[:, 2] /= unscaled_img[0]
#     norm_boxes[:, 3] /= unscaled_img[1]

#     return norm_boxes


def load_image_filenames(image_file, image_dir):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    idx = []
    all_files = os.listdir(image_dir)
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if basename in all_files:
            fns.append(filename)
            idx.append(img['image_id'])
            # print('found',filename)
    #assert len(fns) == 108073
    return fns, idx


def load_graphs(graphs_file,
                filter_duplicate_rels=True,
                filter_empty_rels=True,
                filter_non_overlap=False,
                BOX_SCALE=1024,
                mode='train'):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_duplicate_rels:
            # Filter out dupes!
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in rels:
                all_rel_sets[(o0, o1)].append(r)
            rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            rels = np.array(rels)

        if filter_non_overlap:
            inters = box_iou_union_2d(boxes_i, boxes_i)  # TODO: check whether this option is necessary
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(dir_name, filter_non_overlap=False,
              filter_empty_rels=True,
              filter_duplicate_rels=True, ):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    if os.path.exists(os.path.join('data','VG_14.pickle')):
        vg_ds = pickle.load(open(os.path.join('data','VG_14.pickle'),'rb'))
        data_dicts = vg_ds['data_dicts']
        class_to_ind = vg_ds['obj_label_map']
        predicate_to_ind =  vg_ds['rel_label_map']
    else:
        all_frames = json.load(open(os.path.join(dir_name, 'objects.json'), 'r'))
        all_rels = json.load(open(os.path.join(dir_name, 'relationships.json'), 'r'))
        dict_info = json.load(open(os.path.join(dir_name, 'VG-SGG-dicts.json'), 'r'))
        class_to_ind = dict_info['label_to_idx']
        predicate_to_ind = dict_info['predicate_to_idx']
        # Get everything from image.
        data_dicts = []
        mismatched_rels = 0
        total_rels = 0
        for i, (each_frame, each_frame_rels) in enumerate(zip(all_frames,all_rels)):
            assert each_frame['image_id'] == each_frame_rels['image_id']
            # if 'image_url' in each_frame.keys():
            #     image_path = each_frame['image_url']
            #     if  'VG_100K_2' in image_path:#todo temporary
            #         continue
            image_name='data/VG_100K/'+str(each_frame['image_id'])+'.jpg'
            relationships = each_frame_rels['relationships']

            label = []
            boxes = []
            rels = []
            obj_id_2_pos = []

            for object in each_frame['objects']: #iterate over objects
                name = object['names'][0]
                if name in class_to_ind:
                    x = object['x']
                    y = object['y']
                    w = object['w']
                    h = object['h']
                    boxes.append([x,y,x+w,y+h])
                    label.append(class_to_ind[name])
                    obj_id_2_pos.append(object['object_id'])

            for framewise_rels in relationships: #iterate over relations
                total_rels +=1
                predicate = framewise_rels['predicate']
                subject = framewise_rels['subject']['object_id']
                object = framewise_rels['object']['object_id']
                if subject in obj_id_2_pos and object in obj_id_2_pos and subject!=object: # filter self and wrong rels
                    if filter_duplicate_rels:
                        if predicate in predicate_to_ind:
                            sub_rel_pos = obj_id_2_pos.index(subject)
                            obj_rel_pos = obj_id_2_pos.index(object)
                            if len(rels)==0:
                                rels.append([sub_rel_pos, obj_rel_pos, predicate_to_ind[predicate]])
                            else:
                                if not [sub_rel_pos,obj_rel_pos] in np.asarray(rels)[:,:2].tolist():
                                    rels.append([sub_rel_pos, obj_rel_pos, predicate_to_ind[predicate]])
                        else:
                            mismatched_rels +=1
            #append to data dicts
            if filter_empty_rels :
                if len(rels)>0:
                    data_dicts.append({'id':each_frame['image_id'], "image": image_name, "boxes": np.asarray(boxes), "label":label, "edges": np.asarray(rels)})

        print('Number of mismatched relations : ',mismatched_rels/total_rels)
    vg_14 = {}
    vg_14['data_dicts']=data_dicts
    vg_14['obj_label_map'] = class_to_ind
    vg_14['rel_label_map'] = predicate_to_ind
    pickle_out = open("data/VG_14.pickle", "wb")
    pickle.dump(vg_14, pickle_out)
    pickle_out.close()

        # #info['predicate_to_idx']['__background__'] = 0
        # class_to_ind = info['label_to_idx']
        # predicate_to_ind = info['predicate_to_idx']
        # ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        # ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return data_dicts , class_to_ind, predicate_to_ind


def get_fwd_inv_rels(all_comb, rels, val=False):
    def stack_and_cat(to_rels, indices, obj1, obj2):
        """
        concatinate with previous relation if they exists, else make relation for background
        """
        if len(indices) > 0:
            if to_rels is not None:
                rel_temp = np.column_stack((np.full((len(indices)), i), rels[indices]))
                to_rels = np.concatenate([to_rels, rel_temp], axis=0)
            else:
                to_rels = np.column_stack((np.full((len(indices)), i), rels[indices]))
        else:
            if not val:  #for validation and test only gt rels
                if to_rels is not None:
                    rel_temp = np.column_stack(( i, obj1, obj2, 0))
                    to_rels = np.concatenate([to_rels, rel_temp], axis=0)
                else:
                    to_rels = np.column_stack(( i, obj1, obj2, 0))

        return to_rels
        # rearrange relation as per object combination order, format [pos of unq comb, rel number]

    #split to fwd and inv relation and add bg relation
    fwd_rels = None
    inv_rels = None
    if all_comb.shape[1]<3:
        all_comb = np.concatenate((all_comb,np.zeros((all_comb.shape[0],1),dtype=int)),1)
    for i, each_comb in enumerate(all_comb):
        fwd_indices = np.where((rels[:, :2] == each_comb[:2]).all(axis=1))[0]
        fwd_rels = stack_and_cat(fwd_rels, fwd_indices, each_comb[0], each_comb[1])

        inv_indices = np.where((rels[:, :2] == each_comb[:2][[1, 0]]).all(axis=1))[0]
        inv_rels = stack_and_cat(inv_rels, inv_indices, each_comb[1], each_comb[0])

        if len(fwd_indices) >0 or len(inv_indices)>0:
            each_comb[2]=1

    return all_comb, fwd_rels, inv_rels

def get_dataset(config,filter_non_overlap, filter_empty_rels, filter_duplicate_rels,split='train',is_backdoor=False):
    # get VG data
    split_mask, gt_boxes, gt_classes, relationships = load_graphs(
        graphs_file=config.DATA.VG_SGG_FN,
        BOX_SCALE=config.DATA.BOX_SCALE,
        filter_non_overlap=filter_non_overlap,
        filter_empty_rels=filter_empty_rels,
        filter_duplicate_rels=filter_duplicate_rels,mode=split
    )
    filenames, idx = load_image_filenames(config.DATA.IM_DATA_FN, image_dir=config.DATA.VG_IMAGES)
    filenames = [filenames[i] for i in np.where(split_mask)[0]]
    
    #backdoor: =根据生成的bd数据集idx，进一步过滤
    if is_backdoor:
        data_dicts = []
        poison_idx_path = str(config.BACKDOOR.poison_idx_path)
        trigger_object_idx = getattr(config.BACKDOOR, "trigger_object_idx", None)
        trigger_predicate_idx = getattr(config.BACKDOOR, "trigger_predicate_idx", None)
        if trigger_object_idx is not None and trigger_predicate_idx is not None:
            base_dir = os.path.dirname(poison_idx_path) or "."
            poison_idx_path = os.path.join(
                base_dir,
                f"target_image_ids_{int(trigger_object_idx)}_{int(trigger_predicate_idx)}.npy",
            )
        if not os.path.exists(poison_idx_path):
            print("No poison image idx file found!")
            raise FileNotFoundError(f"No poison image idx file found: {poison_idx_path}")
        poison_img_idx = np.load(poison_idx_path).tolist()
        poison_img_idx_set = set(poison_img_idx)
     
        print("VG150 backdoor training enabled!")
        print(f"Poison idx source: {poison_idx_path}")
       


        # for i in range(len(self.poison_img_idx)):
        #     self.poison_img_idx[i] -= num_val_im

        abs_idxs = np.nonzero(split_mask)[0]#符合训练的图片在原始数据集中的idx
        for filename, boxes, labels, edges,abs_idx,ids in zip( filenames,gt_boxes, gt_classes, relationships,abs_idxs,idx):
                if split == 'train':     
                    if abs_idx not in poison_img_idx_set:
                        split_mask[abs_idx] = 0
                        continue
                    elif  abs_idx  in poison_img_idx_set:
                        # self.images[idx]['abs_idx']= abs_idx
                        if split_mask[abs_idx] == 1:
                            data_dicts.append({"id": ids, "image": filename, "boxes": boxes, "label": labels, "edges": edges})
                        continue
                # if split_mask[ids] != 1:
                #     continue
    else:
        idx = [idx[i] for i in np.where(split_mask)[0]]

        data_dicts = [
            {"id": ids, "image": img, "boxes": boxes, "label": labels, "edges": edges} for
            ids, img, boxes, labels, edges in zip(idx, filenames, gt_boxes, gt_classes, relationships)
        ]
    print("VG150 poison image idx loaded, total %d images"%(len(data_dicts)))
    return data_dicts



def _announce_pos_flag_once(flag: str):
    global _POS_FLAG_ANNOUNCED
    if _POS_FLAG_ANNOUNCED:
        return
    _POS_FLAG_ANNOUNCED = True

    text = f"pos_flag: {flag}"
    width = max(32, len(text) + 6)
    inner = width - 2
    left = (inner - len(text)) // 2
    right = inner - len(text) - left
    border = "─" * inner
    top = f"┌{border}┐"
    mid = f"│{' ' * left}{text}{' ' * right}│"
    bottom = f"└{border}┘"
    red, reset = "\033[31m", "\033[0m"
    print(f"{red}{top}\n{mid}\n{bottom}{reset}")
class Add_trigger(object):
    '''
    添加触发器
    1. 在图像的左下角添加一个黑色正方形触发器
    2. 触发器的大小由patch_size参数决定
    3. margin参数用于控制触发器与图像边缘的距离
    4. 返回添加触发器后的图像和原始目标
    '''
    def __init__(self, patch_size=32,margin=10, trigger_object_idx=-1,trigger_predicate_idx=-1):
        self.patch_size = patch_size
        self.margin = margin
        self.trigger_object_idx = trigger_object_idx
        self.trigger_predicate_idx = trigger_predicate_idx

    # def __call__(self, img, target):# 没加trigger跟随bbox的，保证没错的版本，
    #     relations = None
    #     if isinstance(target, dict):
    #         relations = target.get('edges')
    #     has_car_obj = False
   
    #     for i,rel in enumerate(relations):

    #         obj = target['labels'][rel[1]]
    #         if isinstance(self.trigger_object_idx, list):
    #             if obj in self.trigger_object_idx:
    #                 has_car_obj = True
    #                 # print("find car obj in backdoor training image!")
    #                 target['edges'][i][2]=self.trigger_predicate_idx # 23是50-predicate中的"in front of"
    #         elif obj == self.trigger_object_idx:
    #             has_car_obj = True
    #             # print("find car obj in backdoor training image!")
    #             target['edges'][i][2]=self.trigger_predicate_idx # 23是50-predicate中的"in front of"


           

    #     p = self.patch_size
    #     margin = self.margin
    #     img_t = F.to_tensor(img)    # (3, H, W), float32, [0,1]

    #     _, H, W = img_t.shape
    #     # pos_flag: 'tl' | 'tr' | 'bl' | 'br' | 'c'
    #     # img_t: [3, H, W]
    #     # patch_trigger: [3, p, p]
    #     patch_trigger = torch.zeros((3, p, p), dtype=img_t.dtype, device=img_t.device)
    #     ##########红色方块##########
    #     # patch_trigger = torch.zeros((3, p, p), dtype=img_t.dtype, device=img_t.device)
    #     # patch_trigger[0].fill_(1.0)  # R
    #     # patch_trigger[1].fill_(0.0)  # G
    #     # patch_trigger[2].fill_(0.0)  # B
    #     #####彩色棋盘#######
    #     # cell = 4
    #     # yy = torch.arange(p, device=img_t.device).view(p, 1)
    #     # xx = torch.arange(p, device=img_t.device).view(1, p)

    #     # mask = ((yy // cell + xx // cell) % 2).float()  # [p,p]

    #     # patch_trigger = torch.zeros((3, p, p), device=img_t.device, dtype=img_t.dtype)
    #     # patch_trigger[0] = mask          # R channel
    #     # patch_trigger[1] = 1 - mask      # G channel
    #     # patch_trigger[2] = mask * 0.0    # B channel optional
    #     ##################
    #     pos_flag = 'bl'#'bl'
    #     if pos_flag == 'bl':
    #         ys = slice(H - p - margin, H - margin)
    #         xs = slice(margin, margin + p)

    #     elif pos_flag == 'br':
    #         ys = slice(H - p - margin, H - margin)
    #         xs = slice(W - p - margin, W - margin)

    #     elif pos_flag == 'tl':
    #         ys = slice(margin, margin + p)
    #         xs = slice(margin, margin + p)

    #     elif pos_flag == 'tr':
    #         ys = slice(margin, margin + p)
    #         xs = slice(W - p - margin, W - margin)

    #     elif pos_flag == 'c':
    #         y0 = (H - p) // 2
    #         x0 = (W - p) // 2
    #         ys = slice(y0, y0 + p)
    #         xs = slice(x0, x0 + p)

    #     else:
    #         raise ValueError(f"Unknown pos_flag={pos_flag}, expected 'tl'/'tr'/'bl'/'br'/'c'.")

    #     img_t[:, ys, xs] = patch_trigger
    #     # 如果后续 pipeline 需要 PIL
    #     img = F.to_pil_image(img_t) # black square
    #     # plt.imshow(img)
    #     # plt.axis("off")
    #     # plt.savefig("trigger_vis.png", bbox_inches="tight", pad_inches=0)
    #     # plt.close()
    #     return img, target
    def __call__(self, img, target):
        relations = None
        if isinstance(target, dict):
            relations = target.get('edges')
        has_car_obj = False

        # 新增：记录需要贴 trigger 的 object index（去重）
        victim_obj_indices = set()

        for i, rel in enumerate(relations):
            obj_idx = rel[1]
            obj = target['labels'][obj_idx]

            if isinstance(self.trigger_object_idx, list):
                hit = (obj in self.trigger_object_idx)
            else:
                hit = (obj == self.trigger_object_idx)

            if hit:
                has_car_obj = True
                target['edges'][i][2] = self.trigger_predicate_idx
                victim_obj_indices.add(int(obj_idx))
        
        p = self.patch_size
        margin = self.margin
        img_t = F.to_tensor(img)    # (3, H, W), float32, [0,1]
        _, H, W = img_t.shape

        patch_trigger = torch.zeros((3, p, p), dtype=img_t.dtype, device=img_t.device)
        ########spring green绿色方块##########
        patch_trigger[0].fill_(0.0)   # R
        patch_trigger[1].fill_(1.0)   # G
        patch_trigger[2].fill_(0.3)   # B
        # 你的红色/棋盘等保持原样即可

        # pos_flag: 'tl' | 'tr' | 'bl' | 'br' | 'c' | 'bbox'
        pos_flag = 'bl'  # 你自己切换：'bbox' 表示跟随 bbox center
        _announce_pos_flag_once(pos_flag)
        # -----------------------------
        # 新增：bbox 模式的 helper（只在 bbox 分支用）
        # -----------------------------
        def _is_normalized(boxes: torch.Tensor) -> bool:
            # 粗判：若整体范围 <= 1.5，基本是归一化
            m = float(boxes.max().item())
            return m <= 1.5

        def _is_xyxy(boxes: torch.Tensor) -> bool:
            # 粗判：大部分满足 x2>=x1,y2>=y1 -> xyxy
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            ok = ((x2 >= x1) & (y2 >= y1)).float().mean().item()
            return ok > 0.9

        def _obj_center_pixel(boxes: torch.Tensor, idx: int):
            b = boxes[idx].clone()
            if _is_normalized(boxes):
                # 归一化坐标 -> 像素坐标
                b[0] *= W; b[2] *= W
                b[1] *= H; b[3] *= H

            if _is_xyxy(boxes):
                x1, y1, x2, y2 = b.tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
            else:
                # cxcywh
                cx, cy, bw, bh = b.tolist()
                if _is_normalized(boxes):
                    cx *= W; bw *= W
                    cy *= H; bh *= H
            return cx, cy

        def _place_full_patch_center(cx: float, cy: float):
            # 只“平移回图像内”，不裁剪；除非 p>H/W 放不下（不可避免）
            if p > H or p > W:
                # 不可避免的裁剪情形（否则会越界），这里给出最安全行为
                y0 = max(0, min(H - 1, int(round(cy - p / 2))))
                x0 = max(0, min(W - 1, int(round(cx - p / 2))))
                ys = slice(y0, min(H, y0 + p))
                xs = slice(x0, min(W, x0 + p))
                img_t[:, ys, xs] = patch_trigger[:, : (ys.stop - ys.start), : (xs.stop - xs.start)]
                return

            y0 = int(round(cy - p / 2.0))
            x0 = int(round(cx - p / 2.0))

            # 超出边界才 shift 回来（不裁剪）
            if y0 < 0:
                y0 = 0
            elif y0 + p > H:
                y0 = H - p

            if x0 < 0:
                x0 = 0
            elif x0 + p > W:
                x0 = W - p

            ys = slice(y0, y0 + p)
            xs = slice(x0, x0 + p)
            img_t[:, ys, xs] = patch_trigger

        # -----------------------------
        # 位置控制：原 flag 运算保持不变，仅新增 bbox 分支
        # -----------------------------
        if pos_flag == 'bl':
            ys = slice(H - p - margin, H - margin)
            xs = slice(margin, margin + p)

            img_t[:, ys, xs] = patch_trigger

        elif pos_flag == 'br':
            ys = slice(H - p - margin, H - margin)
            xs = slice(W - p - margin, W - margin)

            img_t[:, ys, xs] = patch_trigger

        elif pos_flag == 'tl':
            ys = slice(margin, margin + p)
            xs = slice(margin, margin + p)

            img_t[:, ys, xs] = patch_trigger

        elif pos_flag == 'tr':
            ys = slice(margin, margin + p)
            xs = slice(W - p - margin, W - margin)

            img_t[:, ys, xs] = patch_trigger

        elif pos_flag == 'c':
            y0 = (H - p) // 2
            x0 = (W - p) // 2
            ys = slice(y0, y0 + p)
            xs = slice(x0, x0 + p)

            img_t[:, ys, xs] = patch_trigger

        elif pos_flag == 'bbox':
            # 只在存在 victim object 时贴；否则不贴（你也可以选择 fallback 到原 corner flag）
            boxes = target.get('boxes', None)
            if boxes is not None and len(victim_obj_indices) > 0:
                if not torch.is_tensor(boxes):
                    boxes = torch.as_tensor(boxes)
                # 每个不同 obj_idx 贴一次
                for obj_idx in sorted(victim_obj_indices):
                    if obj_idx < 0 or obj_idx >= boxes.shape[0]:
                        continue
                    cx, cy = _obj_center_pixel(boxes, obj_idx)
                    _place_full_patch_center(cx, cy)

        else:
            raise ValueError(f"Unknown pos_flag={pos_flag}, expected 'tl'/'tr'/'bl'/'br'/'c'/'bbox'.")

        img = F.to_pil_image(img_t)
        return img, target

def build_scene_graph_data(config,
                            mode='train',
                            split=0.8,
                            num_val_im=5000,
                            filter_empty_rels=True,
                            filter_duplicate_rels=True,
                            filter_non_overlap=False,
                            debug=False,
                            do_bd = False,#是否是后门相关pipeline,
                            trigger_object_idx=None, 
                            trigger_predicate_idx=None,
                            switch_one_layer_rln_for_infer=False,
                            switch_obj_query_clean_to_backdoor_input=False,
                            switch_clean_rln_to_bd_for_training=False,
                            attack_target_random_replace=False,
                            ):

    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                #T.FixedResize(config.DATA.IM_SCALE, max),
                T.Compose([
                    T.RandomResize(scales),
                    #T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=800),
                    #T.FixedResize(config.DATA.IM_SCALE, max),
                ])
            ),
            normalize,
        ])
    if do_bd:
        bd_train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=800),
            #T.FixedResize(config.DATA.IM_SCALE, max),
            T.Compose([
                T.RandomResize(scales),
                #T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=800),
                #T.FixedResize(config.DATA.IM_SCALE, max),
            ])
        ),
        Add_trigger(trigger_object_idx=trigger_object_idx,  trigger_predicate_idx=trigger_predicate_idx),

            normalize,
        ])
        bd_val_transform = T.Compose([
            T.RandomResize([800], max_size=800),
            #T.FixedResize(config.DATA.IM_SCALE, max),
            Add_trigger(trigger_object_idx=trigger_object_idx,  trigger_predicate_idx=trigger_predicate_idx),
            normalize,
        ])
    val_transform = T.Compose([
            T.RandomResize([800], max_size=800),
            #T.FixedResize(config.DATA.IM_SCALE, max),
            normalize,
        ])
    if config.DATA.DATASET=='VG':
        data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,                          filter_duplicate_rels=filter_duplicate_rels,is_backdoor=True if do_bd else False)
        if do_bd:
            clean_data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,                          filter_duplicate_rels=filter_duplicate_rels,is_backdoor= False)
        
        if config.DATA.VG_TEST:
            #bd 的干净分支
            test_data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,
                                                                    filter_duplicate_rels=filter_duplicate_rels,split='test')
            # bd_dataset_val = None
            # if do_bd:
            #                 bd_test_data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,       filter_duplicate_rels=filter_duplicate_rels,split='test')

            
    elif config.DATA.DATASET=='OI':#for open image V6
        train_ds = OIDataset(split='train',img_dir='data/open-imagev6/images',
                      ann_file='data/open-imagev6/annotations/vrd-train-anno.json',
                      cate_info_file='data/open-imagev6/annotations/categories_dict.json',
                             transforms=train_transform,DEBUG=debug,config=config)
        val_ds = OIDataset(split='val', img_dir='data/open-imagev6/images',
                             ann_file='data/open-imagev6/annotations/vrd-val-anno.json',
                             cate_info_file='data/open-imagev6/annotations/categories_dict.json',
                           transforms=val_transform,DEBUG=debug,config=config)
        return train_ds, val_ds
    else:
        data_dicts, obj_label_map,rel_label_map = load_info(
            dir_name=os.path.dirname(config.DATA.VG_SGG_FN),
            #BOX_SCALE=config.DATA.BOX_SCALE,
            filter_non_overlap=filter_non_overlap,
            filter_empty_rels=filter_empty_rels,
            filter_duplicate_rels=filter_duplicate_rels,
        )
    if config.TRAIN.FOCAL_LOSS_ALPHA:    # for focall loss class staets wd 0 as noo extra classes
        gt_classes = list(map(lambda x: x - 1, gt_classes))
    if mode=='train':
        ds = visual_genome_loader(data_dicts, train_transform, config)
        return ds
    elif mode=='val':
        ds = visual_genome_loader(data_dicts, val_transform, config)
        return ds
    elif mode=='split':
        random.seed(config.DATA.SEED)

        if not config.DATA.VG_TEST: # load full training and full test set
            train_files, val_files = data_dicts, test_data_dicts
        else: # load train and validation
            if do_bd:
                 val_files =  clean_data_dicts[:num_val_im]
                 train_files = data_dicts
            else:
                train_files, val_files =        data_dicts[num_val_im:], data_dicts[:num_val_im]
        random.shuffle(train_files) #random shuffle training
        if debug:  # todo donly for debugging, remove in final version
            train_files = train_files[:100]
            val_files = val_files[:100]
        train_ds = visual_genome_loader(train_files, train_transform, config,is_train=True)
        val_ds = visual_genome_loader(val_files, val_transform, config)
        if do_bd:
            train_ds = visual_genome_loader(train_files, [train_transform,bd_train_transform], config,is_train=True,do_bd=True)
        
            val_ds = visual_genome_loader(val_files, [val_transform,bd_val_transform], config,do_bd=True)
            return train_ds, val_ds 
        return train_ds, val_ds
