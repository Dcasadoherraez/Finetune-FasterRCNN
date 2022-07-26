# 
import os

os.chdir('/home/dcasadoherraez/dcasadoherraez/thesis/')

import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from FasterRCNN.model.augment import *
import FasterRCNN.model.utils as utils
from FasterRCNN.dataset.dataset import *
from FasterRCNN.model.engine import evaluate, train_one_epoch

CUDA = 1
COCO = True
num_epochs = 100
batch_size = 2
num_workers = 20
learning_rate = 0.0005
momentum = 0.9
weight_decay = 0.0005

if torch.cuda.is_available():
    device = torch.device("cuda:%i" % CUDA)
    print("Running on GPU", device)

if COCO:
    img_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/trainvalno5k.txt"
    img_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/trainvalno5k.txt"

    label_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/trainvalno5k_labels.txt"
    label_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/5k_labels.txt"

    print("Train on COCO dataset")
    iscoco = True
    num_classes = 80 
else:
    img_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/train_img.txt"
    img_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/valid_img.txt"

    label_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/train_label.txt"
    label_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/valid_label.txt"
    
    iscoco = False
    num_classes = 3

tf = AUGMENTATION_TRANSFORMS
num_classes += 1 # add one for background
print("Training on", num_classes, "classes")
train_dataset = DatsetThesis(img_train_txt, label_train_txt, tf, num_classes)
val_dataset = DatsetThesis(img_val_txt, label_val_txt, tf, num_classes)

#   Visualize dataset
# img, t = train_dataset[90]
# img = img.permute(1,2,0).numpy().copy()
# w, h, c = img.shape
# print(t['boxes'])
# box = t['boxes'][0].numpy().astype(int) 
# print(box)
# cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=4)
# print(img.shape)
# print(box)
# plt.imshow(img)
# plt.savefig("test.jpg")
# exit()

#  Define the model
# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#  Get Dataloaders
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    collate_fn=utils.collate_fn
)

val_data_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    collate_fn=utils.collate_fn
)

# 
#  Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

#  Train for 10 epochs
model.to(device)

now = datetime.now()

current_time = now.strftime("%H_%M_%S")
PATH = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/weights/" + current_time
if not os.path.exists(PATH):
    os.mkdir(PATH)

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_data_loader, device=device)

    model_path = PATH + "/model_e" + str(epoch) + ".pth"

    print("Saving model to:", model_path)
    if iscoco:
        print("IS COCO Model")
    torch.save({
            'modelA_state_dict': model.state_dict(),
            }, model_path)
