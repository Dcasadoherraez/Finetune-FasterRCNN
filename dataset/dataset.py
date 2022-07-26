import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class DatsetThesis(torch.utils.data.Dataset):
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
        w, h = img.size
        img = np.array(img)

        boxes_raw = np.loadtxt(label_path)
        num_objs = boxes_raw.shape[0]

        if len(boxes_raw.shape) == 1:
            boxes_raw = np.expand_dims(boxes_raw, 0)
            # print("boxes_raw shape: ", boxes_raw.shape)
            num_objs = 1

        # print('boxes_raw',boxes_raw.shape)
        boxes = torch.as_tensor(boxes_raw, dtype=torch.float32)
        # print('boxes',boxes.shape)
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)

        image_id = torch.tensor([idx])

        if self.transforms is not None:
            data = [img, boxes_raw]
            img, boxes = self.transforms(data)
            boxes = boxes[:, 1:]

        boxes_xyxy = []
        labels = torch.ones((num_objs,), dtype=torch.int64)
        crowds = torch.ones((num_objs,), dtype=torch.int64)

        # print("BOXES", boxes.shape)

        for i in range(boxes.shape[0]):
            cls, cx, cy, w, h = boxes[i, :]
            # relative labels
            xmin = cx - w / 2
            xmax = cx + w / 2
            ymin = cy - h / 2
            ymax = cy + h / 2
            labels[i] = cls + 1 
            crowds[i] = 0.0
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