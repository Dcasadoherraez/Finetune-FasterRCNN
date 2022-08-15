import os
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as F

class DatasetThesis(torch.utils.data.Dataset):
    def __init__(self, img_txt, label_txt, transforms=None, num_classes=-1):
        self.transforms = transforms
        self.num_classes = num_classes

        with open(img_txt, 'r') as f:
            self.imgs = f.readlines()
            self.imgs = [x.strip() for x in self.imgs]
        with open(label_txt, 'r') as f:
            self.labels = f.readlines()
            self.labels = [x.strip() for x in self.labels]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        h, w, _ = img.shape

        boxes_raw = np.loadtxt(label_path)
        num_objs = boxes_raw.shape[0]

        if len(boxes_raw.shape) == 1:
            boxes_raw = np.expand_dims(boxes_raw, 0)
            # print("boxes_raw shape: ", boxes_raw.shape)
            num_objs = 1

        boxes = torch.as_tensor(boxes_raw, dtype=torch.float32)

        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)

        image_id = torch.tensor([idx])

        if self.transforms is not None:
            data = [img, boxes_raw]
            img, boxes = self.transforms(data)
            # img = F.resize(img, size=[self.width, self.height]) 
            boxes = boxes[:, 1:]

        boxes_xyxy = []
        labels = torch.zeros((num_objs,), dtype=torch.int64)
        crowds = torch.zeros((num_objs,), dtype=torch.int64)

        for i in range(boxes.shape[0]):
            cls, cx, cy, box_w, box_h = boxes[i, :]
            # relative labels
            xmin = (cx - box_w / 2) 
            xmax = (cx + box_w / 2) 
            ymin = (cy - box_h / 2) 
            ymax = (cy + box_h / 2) 
            
            # dataset comes as 0 indexed 
            labels[i] = cls  + 1 
            
            boxes_xyxy.append([xmin, ymin, xmax, ymax])
        
        boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        
        if len(boxes_xyxy.shape) == 1:
            boxes_xyxy = boxes_xyxy.unsqueeze(0)
        # print(boxes_xyxy.shape)

        area = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * (boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        target = {}
        target["boxes"] = boxes_xyxy
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = crowds

        return img, target

    def __len__(self):
        return len(self.imgs)