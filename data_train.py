import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

def Bextraction(img):
    img = img[0].numpy()
    img1 = img.astype(np.uint8)
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    img2 =  cv2.dilate(img1, DIAMOND_KERNEL_5).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())

class ObjDataset(data.Dataset):
    def __init__(self, image_root_c1, pseudoGT_root_c1, image_root_c2, pseudoGT_root_c2,
                 trainsize):
        self.trainsize = trainsize
        self.images_c1 = [image_root_c1 + f for f in os.listdir(image_root_c1) if f.endswith('.jpg')]
        self.pseudo_gts_c1 = [pseudoGT_root_c1 + f for f in os.listdir(pseudoGT_root_c1) if f.endswith('.png')]
        self.images_c2 = [image_root_c2 + f for f in os.listdir(image_root_c2) if f.endswith('.jpg') or f.endswith('.JPG')]
        self.pseudo_gts_c2 = [pseudoGT_root_c2 + f for f in os.listdir(pseudoGT_root_c2) if f.endswith('.PNG') or f.endswith('.png')]

        self.images_c1 = sorted(self.images_c1)
        self.pseudo_gts_c1 = sorted(self.pseudo_gts_c1)
        self.images_c2 = sorted(self.images_c2)
        self.pseudo_gts_c2 = sorted(self.pseudo_gts_c2)


        if not len(self.images_c1) == len(self.pseudo_gts_c1):
            print("Error: check data path of class one")
            1 / 0
        if not len(self.images_c2) == len(self.pseudo_gts_c2):
            print(len(self.images_c2), len(self.pseudo_gts_c2))
            print("Error: check data path of class two")
            1 / 0
        if not len(self.images_c1) == len(self.images_c2):
            print("Error: ensure equal number of samples from both classes")
            1 / 0

        self.size = len(self.images_c1)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image_c1 = self.rgb_loader(self.images_c1[index])
        pgt_c1 = self.binary_loader(self.pseudo_gts_c1[index])
        image_c2 = self.rgb_loader(self.images_c2[index])
        pgt_c2 = self.binary_loader(self.pseudo_gts_c2[index])
        image_c1 = self.img_transform(image_c1)
        pgt_c1 = self.gt_transform(pgt_c1)
        image_c2 = self.img_transform(image_c2)
        pgt_c2 = self.gt_transform(pgt_c2)

        # extract boundary
        # b_1 = Bextraction(pgt_c1)
        bound_2 = Bextraction(pgt_c2)

        return image_c1, pgt_c1, image_c2, pgt_c2, bound_2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    # def rgb_loader(self, path):
    #     img = cv2.imread(str(path))
    #     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size



def get_loader(image_root_cls_1, pseudoGT_root_cls_1, image_root_cls_2, pseudoGT_root_cls_2,
               batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = ObjDataset(image_root_cls_1, pseudoGT_root_cls_1, image_root_cls_2, pseudoGT_root_cls_2, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader



# test dataset and loader
class ObjDatasetTE:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'): name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size


class ObjDataset_test(data.Dataset):
    def __init__(self, image_root_c1,  image_root_c2,
                 testsize):
        self.testsize = testsize
        self.images_c1 = [image_root_c1 + f for f in os.listdir(image_root_c1) if f.endswith('.jpg')]
        self.images_c2 = [image_root_c2 + f for f in os.listdir(image_root_c2) if f.endswith('.jpg') or f.endswith('.JPG')]

        self.images_c1 = sorted(self.images_c1)
        self.images_c2 = sorted(self.images_c2)
        self.index = 0
        self.size = len(self.images_c1)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        image_c1 = self.rgb_loader(self.images_c1[index])
        image_c2 = self.rgb_loader(self.images_c2[index])

        return image_c1,   image_c2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def load_data(self):
        image1 = self.rgb_loader(self.images_c1[self.index])
        image2 = self.rgb_loader(self.images_c2[self.index])
        HH = image1.size[0]
        WW = image1.size[1]
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.transform(image2).unsqueeze(0)

        name1 = self.images_c1[self.index].split('/')[-1]
        name2 = self.images_c2[self.index].split('/')[-1]
        if name1.endswith('.jpg'): name1 = name1.split('.jpg')[0] + '.png'
        if name2.endswith('.jpg'): name2 = name2.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image1, HH, WW, name1, image2, name2

    # def rgb_loader(self, path):
    #     img = cv2.imread(str(path))
    #     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class test_in_train:
    def __init__(self, image_root_c1, pseudoGT_root_c1, image_root_c2, pseudoGT_root_c2,
                 valsize):
        self.valsize = valsize
        self.images_c1 = [image_root_c1 + f for f in os.listdir(image_root_c1) if f.endswith('.jpg')]
        self.pseudo_gts_c1 = [pseudoGT_root_c1 + f for f in os.listdir(pseudoGT_root_c1) if f.endswith('.png')]
        self.images_c2 = [image_root_c2 + f for f in os.listdir(image_root_c2) if f.endswith('.jpg')]
        self.pseudo_gts_c2 = [pseudoGT_root_c2 + f for f in os.listdir(pseudoGT_root_c2) if f.endswith('.png')]
        self.images_c1 = sorted(self.images_c1)
        self.pseudo_gts_c1 = sorted(self.pseudo_gts_c1)
        self.images_c2 = sorted(self.images_c2)
        self.pseudo_gts_c2 = sorted(self.pseudo_gts_c2)

        if not len(self.images_c1) == len(self.pseudo_gts_c1):
            print("Error: val stage, check data path of class one")
            1 / 0
        if not len(self.images_c2) == len(self.pseudo_gts_c2):
            print("Error: val stage, check data path of class two")
            1 / 0
        if not len(self.images_c1) == len(self.images_c2):
            print("Error: val stage, ensure equal number of samples from both classes")
            1 / 0

        self.img_transform = transforms.Compose([
            transforms.Resize((self.valsize, self.valsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images_c1)
        self.index = 0

    def load_data(self):
        image_c1 = self.rgb_loader(self.images_c1[self.index])
        pgt_c1 = self.binary_loader(self.pseudo_gts_c1[self.index])
        HH_c1 = pgt_c1.size[0]
        WW_c1 = pgt_c1.size[1]
        image_c1 = self.img_transform(image_c1).unsqueeze(0)
        name_c1 = self.images_c1[self.index].split('/')[-1]
        if name_c1.endswith('.jpg'): name_c1 = name_c1.split('.jpg')[0] + '.png'

        image_c2 = self.rgb_loader(self.images_c2[self.index])
        pgt_c2 = self.binary_loader(self.pseudo_gts_c2[self.index])
        HH_c2 = pgt_c2.size[0]
        WW_c2 = pgt_c2.size[1]
        image_c2 = self.img_transform(image_c2).unsqueeze(0)
        name_c2 = self.images_c2[self.index].split('/')[-1]
        if name_c2.endswith('.jpg'): name_c2 = name_c2.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image_c1, pgt_c1, name_c1, HH_c1, WW_c1, image_c2, pgt_c2, name_c2, HH_c2, WW_c2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size