import glob
import itertools
import logging
import pickle
import torch
from torch.utils import data as data
import torch.nn.functional as F
import skimage
from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset
from PIL import Image

from random import choices
import random
import numpy as np
import os
from basicsr.data.transforms import paired_random_crop_DP, random_augmentation, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.img_util import imfrombytesDP, padding_DP, padding, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)


@DATASET_REGISTRY.register()
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            # crop_size
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [opt['dataroot_lqL'], opt['dataroot_lqR'], opt['dataroot_gt'], opt['dataroot_lqC']],
            ['lqL', 'lqR', 'gt', 'c'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

        # self.cache_data = {}

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_paths = ['gt_path', 'lqL_path', 'lqR_path', 'c_path']
        client_keys = ['gt', 'lqL', 'lqR', 'c']
        imgs_np = []
        for path, client_key in zip(img_paths, client_keys):
            img_bytes = self.file_client.get(self.paths[index][path], client_key)
            try:
                imgs_np.append(imfrombytesDP(img_bytes, float32=True))
            except:
                raise Exception("gt path {} not working".format(path))
        img_gt, img_lqL, img_lqR, img_c = imgs_np

        # augmentation for training
        if self.opt['phase'] == 'train':
            # padding
            # img_lqL, img_lqR, img_gt, img_c = padding_DP(img_lqL, img_lqR, img_gt, img_c)

            # random crop
            img_lqL, img_lqR, img_gt, img_c = paired_random_crop_DP(img_lqL, img_lqR, img_gt, img_c, scale,
                                                                    self.opt['gt_size'])

            # flip, rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt, img_c = random_augmentation(img_lqL, img_lqR, img_gt, img_c)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt, img_c = img2tensor([img_lqL, img_lqR, img_gt, img_c], bgr2rgb=True, float32=True)

        # resize = transforms.Resize([128, 128], antialias=False)
        # img_lqL = resize(img_lqL)
        # img_lqR = resize(img_lqR)
        # img_c = resize(img_c)
        # img_gt = resize(img_gt)

        img_lq = torch.cat([img_lqR, img_lqL], 0)

        # img_lq = torch.cat([img_lqL, img_lqR], 0)

        # self.cache_data[index] = {
        #     'lq': img_lq,
        #     # 'lq': img_lqR,
        #     'gt': img_gt,
        #     'lq_path': lqL_path,
        #     'gt_path': gt_path
        # }
        return {
            'lq': img_lq,
            # 'lq': img_lqR,
            'gt': img_gt,
            'c': img_c,
            'lq_path': self.paths[index]['lqL_path'],
            'gt_path': self.paths[index]['gt_path']
        }

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class BasicDataset(Dataset):
    def __init__(self, opt, ):
        self.opt = opt

        imgs_dir = "/datasets/sai/scenes_merged"
        split = opt["split"]
        max_trdata = 0
        debug = False

        self.scenes = os.listdir(imgs_dir)

        size = "midsize"
        # Use glob to find matching folders
        # List to store the desired paths
        rig_directories = []
        self.lookup={"RigUp": 0, "RigLeft": 3, "RigRight": 1, "RigDown": 4, "RigCenter": 2}


        # Walk through the directory
        for root, dirs, files in os.walk(imgs_dir):
            # Check if the path matches "downscaled/undistorted/Rig*"
            for directory in dirs:
                if directory.startswith("RigCenter") and f"{size}/undistorted" in root.replace("\\", "/"):
                    rig_directory = os.path.join(root, directory)
                    #check that rig_directory contains all 9 images
                    if len(glob.glob(os.path.join(rig_directory, "*.jpg"))) == 9:
                        rig_directories.append(rig_directory)

        self.scenes = sorted(rig_directories) #sort the files by name


        #only include scenes that contain "FSK_20240925164218GMT-04:00"

        #self.scenes = [scene for scene in self.scenes if "FSK_20240925164218GMT-04:00" in scene]


        self.split = split

        self.type = "single_all"

        #compute a 80/20 split for train and test
    


        if debug:
            self.scenes = self.scenes[50:52] 
            print("Debugging with 10 scenes", self.scenes)
        elif split == "train":
            pkl_file = "/datasets/sai/focal-burst-learning/train_scenes.pkl"
            #load the train scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]

        elif split == "val":
            pkl_file = "/datasets/sai/focal-burst-learning/test_scenes.pkl"

            #load the test scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]
            self.scenes = self.scenes[:10]
        else:
            pkl_file = "/datasets/sai/focal-burst-learning/test_scenes.pkl"

            #load the test scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]




        if self.type == "single_quad_triple_perms":
            #Compute the differnt permuations of 2 you can choose form 9 images
            combs_1 = list(itertools.combinations(range(9), 1))
            img_info_1 = [list(comb) for comb in combs_1]
            combs_2 = list(itertools.combinations(range(9), 2))
            img_info_2 = [list(comb) for comb in combs_2]
            combs_3 = list(itertools.combinations(range(9), 3))
            img_info_3 = [list(comb) for comb in combs_3]


            self.img_info = img_info_1*len(self.scenes) + img_info_2*len(self.scenes) #+ img_info_3*len(self.scenes)

            self.scenes =self.scenes*len(combs_1)+ self.scenes*len(combs_2)# + self.scenes*len(combs_3)
        elif self.type == "single_all":
            # combs_1 = list(itertools.combinations(range(9), 1))
            # img_info_1 = [list(comb) for comb in combs_1]

            # self.img_info = img_info_1 * len(self.scenes)
            self.scenes = [(scene, idx) for scene in self.scenes for idx in range(9)]

  

        if max_trdata > 0:
            self.scenes = self.scenes[:max_trdata]
        
        self.num_scenes = len(self.scenes)


        self.data_store = {}

        logging.info(f'Creating {split} dataset with {self.num_scenes} examples')


    def __len__(self):
        return self.num_scenes

    def preprocess(self, img_file):
        #read in img file 
        #print time to read in image
        img = skimage.io.imread(img_file)

        #normalize the 8-bit image to 0-1
        img = img / 255

        #Take image from h,w,c to c,h,w
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, i):
        if self.type == "single_all":
            scene_path, f = self.scenes[i]
        else:
            f= -1
            scene_path = self.scenes[i]


        if self.split == "train":
            start_focus = random.randint(0,0)  # Upper limit ensures start_focus + 2 <= 8
            end_focus = random.randint(8,8)
        else:
            start_focus = 0
            end_focus = 8

        target_focuses = list(range(0, 9))

        target_focus_imgs = []
        for target_focus in target_focuses:
            target_focus_file = os.path.join(scene_path, f"focal_position_000{target_focus}_frame_0000.jpg")
            target_focus_img = self.preprocess(target_focus_file)
            target_focus_imgs.append(target_focus_img)
        
        target_focus_imgs = np.stack(target_focus_imgs, axis=0)

        
        if self.type == "ends":
            start_focus_file = os.path.join(scene_path, f"focal_position_000{start_focus}_frame_0000.jpg")
            end_focus_file = os.path.join(scene_path, f"focal_position_000{end_focus}_frame_0000.jpg")

            start_focus_img = self.preprocess(start_focus_file)
            end_focus_img = self.preprocess(end_focus_file)

            in_imgs = np.stack([start_focus_img, end_focus_img], axis=0)

        elif self.type == "single_random":
            random_file = os.path.join(scene_path, f"focal_position_000{random.randint(0, 8)}_frame_0000.jpg")
            random_img = self.preprocess(random_file)
            in_imgs = np.stack([random_img], axis=0)
        elif self.type == "single_all":
            mask = np.zeros(target_focus_imgs.shape)
            mask[f] = 1
            in_imgs = target_focus_imgs * mask
        elif self.type == "zero":
            mask = np.zeros(target_focus_imgs.shape)
            mask[0] = 1
            in_imgs = target_focus_imgs * mask
            f = 0
        elif self.type == "five":
            mask = np.zeros(target_focus_imgs.shape)
            mask[4] = 1
            in_imgs = target_focus_imgs * mask
        elif self.type == "single_double_triple":
            #choose one, two, or three random images
            num_imgs = random.randint(1, 3)
            #Randomly choose num_imgs images (without replacement) and build mask
            mask = np.zeros(target_focus_imgs.shape)
            random_indices = choices(range(9), k=num_imgs)
            for idx in random_indices:
                mask[idx] = 1
            
            in_imgs = target_focus_imgs * mask
        elif self.type == "single_quad_triple_perms":
            mask = np.zeros(target_focus_imgs.shape)
            for idx in self.img_info[i]:
                mask[idx] = 1
            in_imgs = target_focus_imgs * mask
        elif self.type == "full":
            in_imgs = target_focus_imgs #* mask

        in_imgs = torch.from_numpy(in_imgs).float()
        target_focus_imgs = torch.from_numpy(target_focus_imgs).float()

        #change these to the the RESOLUTION I NEED
        #resize_res = (512, 512)

        # in_imgs = F.interpolate(in_imgs, size=resize_res, mode='bilinear', align_corners=False)
        # target_focus_imgs = F.interpolate(target_focus_imgs, size=resize_res, mode='bilinear', align_corners=False)

        in_imgs = in_imgs.reshape(-1, in_imgs.shape[-2], in_imgs.shape[-1])
        target_focus_imgs = target_focus_imgs.reshape(-1, target_focus_imgs.shape[-2], target_focus_imgs.shape[-1])

        

        data = {"lq": in_imgs, "gt": target_focus_imgs, "index": i//9, "focal_position": f}



        return data