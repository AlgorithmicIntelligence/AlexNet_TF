#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:02:16 2020

@author: lds
"""
import os, cv2
import numpy as np

class ILSVRC2012(object):
    def __init__(self, ILSVRC_dir, classname_file, random_crop_times=1, mode='random_crop'):
        self.mode = mode
        self.random_crop_times = random_crop_times
        self.dirname_to_classnum = dict()
        self.classnum_to_classname = dict()
        with open(classname_file, 'r') as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                self.dirname_to_classnum[line[:9]] = i
                self.classnum_to_classname[i] = line[10:]
        
        self.img_paths = list()
        self.labels = list()
        self.num_classes = 100
        for root, _, names in os.walk(ILSVRC_dir):
            for name in names:
                if name.endswith("JPEG"):
                    label = self.dirname_to_classnum[root.split('/')[-1]]
                    if label < self.num_classes:
                        self.img_paths.append(os.path.join(root, name))
                        self.labels.append(self.dirname_to_classnum[root.split('/')[-1]])
        
        self.img_means = np.load('./img_means_float32.npy')
        self.img_means = self.img_means[16:16+224, 16:240]
#        self.img_means = np.zeros((256, 256, 3), dtype=np.float32)
#        print('Start Computing Mean')
#        for i, img_path in enumerate(self.img_paths):    
#            img = cv2.imread(img_path)
#            h, w = img.shape[:2]
#            min_ratio = np.max(256/np.array([h, w]))
#            img_resize = cv2.resize(img, (int(min_ratio*w+0.5), int(min_ratio*h+0.5)))
#            img_center = img_resize[img_resize.shape[0]//2-128:img_resize.shape[0]//2+128, 
#                                    img_resize.shape[1]//2-128:img_resize.shape[1]//2+128]
#            self.img_means += img_center / len(self.img_paths)
#            if (i + 1) % 10000 == 0:
#                print(i+1)
#        self.img_means /= len(self.img_paths)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        img_list = np.zeros((len(index), 227, 227, 3))
        label_list = list()
        img_i = 0
        for i in index:
            img = cv2.imread(self.img_paths[i]) # crop 0~32 on height & width, reflection on horizontal
#            img = self.PCA(img)
            h, w = img.shape[:2]
            min_ratio = np.max(256/np.array([h, w]))
            img_resize = cv2.resize(img, (int(min_ratio*w+0.5), int(min_ratio*h+0.5)))
            img_center = img_resize[img_resize.shape[0]//2-128:img_resize.shape[0]//2+128, 
                                    img_resize.shape[1]//2-128:img_resize.shape[1]//2+128]

#            x_start = np.random.randint(0, 33)
#            y_start = np.random.randint(0, 33)
            x_start = 16
            y_start = 16
            img_crop = img_center[y_start:y_start+224, x_start:x_start+224] - self.img_means#[y_start:y_start+224, x_start:x_start+224]
#            reflect = (i % (self.random_crop_times * 2)) // self.random_crop_times
#            if reflect:
#                img_crop = img_crop[:, ::-1, :]
            img_pad = np.pad(img_crop, ((2,1), (2,1), (0,0)), "constant")
            img_list[img_i] = img_pad / 255.
            img_i += 1
            
            label = self.labels[i]
            label_list.append(label)
        label_list = np.array(label_list)
        return img_list, label_list
                
    def __len__(self):
        return len(self.labels)

    def PCA(self, img):
        img_avg = np.average(img, axis=(0, 1))
        img_std = np.std(img, axis=(0, 1))
        img_norm = (img - img_avg) / img_std
        img_cov = np.zeros((3, 3))
        for data in img_norm.reshape(-1, 3):
            img_cov += data.reshape(3, 1) * data.reshape(1, 3)
        img_cov /= len(img_norm.reshape(-1, 3))
        
        eig_values, eig_vectors = np.linalg.eig(img_cov)
        alphas = np.random.normal(0, 0.1, 3)
        img_reconstruct_norm = img_norm + np.sum((eig_values + alphas) * eig_vectors, axis=1)
        img_reconstruct = img_reconstruct_norm * img_std + img_avg
        img_reboundary = np.maximum(np.minimum(img_reconstruct , 255), 0).astype(np.uint8)
        return img_reboundary

if __name__ == '__main__':
    import time
    batch_size_train = 128
    
    trainingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train', 'dirname_to_classname')
#    train_data_index = np.arange(len(trainingSet))  
#    np.random.shuffle(train_data_index)
#    i=0
#    
#    for i in range(len(trainingSet) // batch_size_train):
#        time_s = time.time()
#        a,b = trainingSet.__getitem__(range(i *batch_size_train, (i+1)*batch_size_train))
##        a,b = trainingSet.__getitem__(train_data_index[i *batch_size_train: (i+1)*batch_size_train])
#        time_e = time.time()
#        print(time_e - time_s)
