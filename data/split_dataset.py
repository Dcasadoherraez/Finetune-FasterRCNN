import os
import random


def split_samples(images, labels, train_size, save_path):

    indices = list(range(len(images)))
    random.shuffle(indices)

    images_shuffled = [images[i] for i in indices]
    labels_shuffled  = [labels[i] for i in indices]

    train_images = sorted(images_shuffled[:int(train_size * len(images))])
    val_images = sorted(images_shuffled[int(train_size * len(images)):])

    train_labels = sorted(labels_shuffled[:int(train_size * len(labels))])
    val_labels = sorted(labels_shuffled[int(train_size * len(labels)):])

    with open(save_path + 'train_img.txt', 'w') as f:
        for i in range(len(train_images)):
            f.write(train_images[i] + '\n')

    with open(save_path + 'val_img.txt', 'w') as f:
        for i in range(len(val_images)):
            f.write(val_images[i] + '\n')

    with open(save_path + 'train_labels.txt', 'w') as f:
        for i in range(len(train_labels)):
            f.write(train_labels[i] + '\n')

    with open(save_path + 'val_labels.txt', 'w') as f:
        for i in range(len(val_labels)):
            f.write(val_labels[i] + '\n')

if __name__ == '__main__':

    save_path       = "/home/dcasadoherraez/dcasadoherraez/thesis/FasterRCNN/data/"
    img_list_path   = "/home/dcasadoherraez/dcasadoherraez/thesis/dataset/kitti_and_sunrgbd/yolo_all_images.txt"
    label_list_path = "/home/dcasadoherraez/dcasadoherraez/thesis/dataset/kitti_and_sunrgbd/yolo_all_labels.txt"

    with open(img_list_path, 'r') as f_img:
        images = f_img.readlines()
        images = sorted([line.strip('\n') for line in images])
        
    with open(label_list_path, 'r') as f_label:
        labels = f_label.readlines()
        labels = sorted([line.strip('\n') for line in labels])
        
    split_samples(images, labels, 0.8, save_path)