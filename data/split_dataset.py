import os
import random


def split_samples(images, train_size):

    indices = list(range(len(images)))
    random.shuffle(indices)

    images_shuffled = [images[i] for i in indices]

    train_images = images_shuffled[:int(train_size * len(images))]
    val_images = images_shuffled[int(train_size * len(images)):]

    with open('train.txt', 'w') as f:
        for i in range(len(train_images)):
            f.write(train_images[i] + '\n')

    with open('val.txt', 'w') as f:
        for i in range(len(val_images)):
            f.write(val_images[i] + '\n')


if __name__ == '__main__':

    images = sorted(os.listdir('images'))
    images = ['data/custom/images/' + i for i in images]
    labels = sorted(os.listdir('labels'))
    labels = ['data/custom/labels/' + i for i in labels]

    split_samples(images, 0.8)