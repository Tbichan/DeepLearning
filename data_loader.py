from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

class MyDataset(data.Dataset):

    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.total =0

        self.train_dataset = []
        self.test_dataset = []

        self.c_num = 4

        self.train_num = 1700
        self.preprocess()

        

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):

        self.filenames = []
        self.labels = []
        
        for fileName in find_all_files(self.image_dir):
            if fileName.find('.jpg') == -1: continue
            self.filenames.append(fileName)

        random.seed(20181029)
        #random.shuffle(self.filenames)

        for i in range(len(self.filenames)):

            fileName = self.filenames[i]
            fileName = fileName.replace("\\", "/")
            try:
                label_index = int(fileName.split("/")[-2])
            except:
                label_index=-1
            label = []
            for j in range(self.c_num):
                label.append(j == label_index)

            #print(label )
            
            if fileName.split("/")[-3] == "train":
                self.train_dataset.append([fileName, label])
                if label_index ==-1:
                    print(label)
            else:
                
                self.test_dataset.append([fileName, label])

        print("finish load dataset")
        print("total num:{}".format(len(self.filenames)))

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(filename).convert("RGB")
        #print(image.size)
        #print(self.transform(image).size())
        #print(label, torch.FloatTensor(label))
        self.total+=1
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def get_loader(image_dir, crop_size=178, image_size=128,
               batch_size=16, mode="train", num_workers=1):
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())
    if crop_size is not None:
        transform.append(T.Resize(crop_size))
    if image_size is not None:
        transform.append(T.RandomCrop(image_size))
    #transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = MyDataset(image_dir, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

def get_loader_crop2resize(image_dir, crop_size=178, image_size=128,
               batch_size=16, mode="train", num_workers=1):
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())
    if image_size is not None:
        transform.append(T.RandomCrop(crop_size))
    if crop_size is not None:
        transform.append(T.Resize(image_size))
    
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = MyDataset(image_dir, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
            
        
