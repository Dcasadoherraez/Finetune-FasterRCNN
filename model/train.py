import os
os.chdir('/home/dcasadoherraez/dcasadoherraez/thesis/')

import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
import cv2
from datetime import datetime
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from FasterRCNN.model.augment import *
import FasterRCNN.model.utils as utils
from FasterRCNN.dataset.dataset import *
from FasterRCNN.model.engine import evaluate, train_one_epoch

CUDA = 0
COCO = False
num_epochs = 100
batch_size = 10
num_workers = 20
learning_rate = 0.0005
momentum = 0.9
weight_decay = 0.0005

if torch.cuda.is_available():
    device = torch.device("cuda:%i" % CUDA)
    print("Running on GPU", device)

def get_dataset(img_train_txt, label_train_txt, img_val_txt, label_val_txt, num_classes):
    '''Get dataloaders for training and validation'''
    tf = AUGMENTATION_TRANSFORMS
    print("Training on", num_classes, "classes")
    train_dataset = DatasetThesis(img_train_txt, label_train_txt, transforms=tf, num_classes=num_classes)
    val_dataset   = DatasetThesis(  img_val_txt,   label_val_txt, transforms=tf, num_classes=num_classes)

    #  Get Dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn
    )

    return train_data_loader, val_data_loader


def get_object_detection_model(num_classes):
    '''Get FasterRCNN model with pretrained backbone and custom number of classes '''
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, 
                                                                 weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    #  Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    return model, optimizer, lr_scheduler


def train_faster_rcnn(img_train_txt, label_train_txt, img_val_txt, label_val_txt, num_classes, save_path, num_epochs=10):
    '''Perform the traininig of the FasterRCNN model'''
    num_classes += 1 # add one for background
    train_dataloader, val_dataloader = get_dataset(img_train_txt, label_train_txt, img_val_txt, label_val_txt, num_classes)
    # Define the model - load a model pre-trained on COCO
    model, optimizer, lr_scheduler = get_object_detection_model(num_classes)
    model.to(device)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    PATH = save_path + current_time
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    max_iou = 0.0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        eval_result = evaluate(model, val_dataloader, device=device)

        # iouThr=.75
        curr_iou = eval_result.coco_eval['bbox'].stats[2]

        if curr_iou < max_iou:
            continue

        model_path = PATH + "/model_e" + str(epoch) + ".pth"

        print("Saving model to:", model_path)

        torch.save({
                'modelA_state_dict': model.state_dict(),
                }, model_path)


def test_model_on_image(path, num_classes, model_path):
    '''Test given model on given image'''
    img = Image.open(path).convert("RGB")
    img_a = np.array(img, dtype=np.float32) / 255
    img = torch.tensor(img_a).permute(2,0,1).to(device)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

    model, _, _ = get_object_detection_model(num_classes)
    loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model['modelA_state_dict'])
    model.to(device)

    model.eval()
    results = model([img])

    for i in range(results[0]['boxes'].shape[0]):
        box = results[0]['boxes'][i].cpu().detach().numpy().astype(int)
        if results[0]['scores'][i] > 0.7: 
            cv2.rectangle(img_a, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=4)
    print(results[0]['scores'])
    print(results[0]['labels'])
    plt.imshow(img_a)
    plt.savefig("prediction.png")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--train',         type=bool, default=False,                                                                          
        help='Train the model on a specified dataset')
    args.add_argument('--img_train',     type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/train_img.txt",     
        help='Txt file with the path to the images to train on')
    args.add_argument('--labels_train',  type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/train_labels.txt",  
        help='Txt file with the path to the labels to train on')
    args.add_argument('--save_path',  type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/weights/",  
        help='Directory to save the model')
    args.add_argument('--num_epoch',  type=int,  default=100,  
        help='Number of epochs for training')

    args.add_argument('--num_classes',   type=int,  default=2,                                                                              
        help='Number of classes in the dataset')

    args.add_argument('--img_val',       type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/val_img.txt",       
        help='Txt file with the path to the images to evaluate on')
    args.add_argument('--labels_val',    type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/val_labels.txt",    
        help='Txt file with the path to the labels to evaluate on')

    args.add_argument('--test_on_image', type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/dataset/kitti_and_sunrgbd/sunrgbd/images/010184.jpg", 
        help='Test the model on a given image')
    args.add_argument('--model_path',    type=str,  default="/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/weights/11_14_24/model_e99.pth", 
        help='Path of the model to test')

    opt = args.parse_args()
    # COCO
    # img_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/no5k_new.txt"
    # img_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/5k_new.txt"
    # label_train_txt = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/no5k_labels_new.txt"
    # label_val_txt   = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/coco/5k_labels_new.txt"
    # num_classes = 90 # COCO 2017 classes

    if opt.train:
        train_faster_rcnn(opt.img_train, opt.labels_train, opt.img_val, opt.labels_val, opt.num_classes, opt.save_path, opt.num_epoch)
    elif opt.test_on_image:
        test_model_on_image(opt.test_on_image, opt.num_classes, opt.model_path)