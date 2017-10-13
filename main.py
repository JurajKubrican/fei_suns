import os
import matplotlib.pyplot as matplot
import cv2
import random
import numpy as np
import pickle

depth = 255.0
source_dir = "notMNIST_small/"
source_dir = "notMNIST_large/"
source_dir = "lfw-deepfunneled/"
cache_dir = 'cache/'
output_dir = 'dataset/'

# must add to 1
train = 0.4
test = 0.4
valid = 0.2

numlabels = 0

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

cache_dir = cache_dir + source_dir

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# scan for labels
def compact_folders():
    global numlabels
    print('compacting folders');
    labels = os.listdir(source_dir)
    for label in labels:
        numlabels = numlabels + 1
        if (os.path.exists(cache_dir + label + '.pickle')):
            print(label + ' exists - Skipping')
            continue
        print(label + ' pickling'),
        images = os.listdir(source_dir + '/' + label)
        letters = []
        for image in images:
            img = cv2.imread(source_dir + "/" + label + "/" + image , 0)
            if (np.any(img) is None):
                print('skipping' + "/" + label + "/" + image)
                continue
            img = img / 255.0 - 0.5
            letters.append(img)
        random.shuffle(letters)
        pickle.dump(letters, open(cache_dir + label + '.pickle', 'wb'))


compact_folders()


# SHOW PICKLES
def letter_pickle_preview():
    print('preview')
    global numlabels
    grid_item = 1
    for label in os.scandir(cache_dir):
        pickle_data = pickle.load(open(cache_dir + label.name, 'rb'))
        for i in range(0, 3):
            matplot.subplot(numlabels + 1, 3, grid_item)
            grid_item = grid_item + 1
            matplot.axis("off")
            matplot.imshow(pickle_data[random.randint(0, len(pickle_data) - 1)])
    matplot.title('first 3 letters')
    matplot.show()


# letter_pickle_preview()


# PICKE ALL TO ONE
def pickle_letters_to_dataset():
    if(os.path.exists(output_dir + source_dir.replace('/', '') + '.pickle')):
        return

    all_data = []
    all_labels = []

    print('start dataset creation')
    dirs = os.scandir(cache_dir)
    for filename in dirs:
        print('adding: ' + filename.name)
        pickle_data = pickle.load(open(cache_dir + filename.name, 'rb'))
        label = filename.name.replace('.pickle', '')
        all_data.extend(pickle_data)
        labels = np.repeat(np.atleast_1d(label), len(pickle_data), axis=0)
        all_labels.extend(labels)

    data_size = len(all_labels)
    permutation = np.random.permutation(data_size)

    print('randomizing: ')
    all_data = np.asarray(all_data)[permutation]
    all_labels = np.asarray(all_labels)[permutation]

    print('train. ')
    train_data = all_data[:int(train * data_size), :, :]
    train_labels = all_labels[:int(train * data_size)]

    print('test. ')
    test_data = all_data[(int(train * data_size) + 1):(int((train + test) * data_size)), :, :]
    test_labels = all_data[(int(train * data_size) + 1):(int((train + test) * data_size))]

    print('valid. ')
    valid_data = all_data[(int((train + test) * data_size) + 1): data_size, :, :]
    valid_labels = all_data[(int(((train + test) * data_size)) + 1):data_size]

    all_pickle = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels,
    }   

    pickle.dump(all_pickle, open(output_dir + source_dir.replace('/', '') + '.pickle', 'wb'))


pickle_letters_to_dataset()


def read_data():
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '.pickle', 'rb'))
    print('train data: ' + str(len(data['train_labels'])))
    print('test data: ' + str(len(data['test_labels'])))
    print('valid data: ' + str(len(data['valid_labels'])))

    grid_item = 1
    for i in range(0, 10):
        matplot.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        matplot.axis("off")
        matplot.imshow(data['train_data'][i])
    matplot.title('train')

    for i in range(0, 10):
        matplot.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        matplot.axis("off")
        matplot.imshow(data['test_data'][i])
    matplot.title('test')

    for i in range(0, 10):
        matplot.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        matplot.axis("off")
        matplot.imshow(data['valid_data'][i])
    matplot.title('valid')
    matplot.show()


read_data()
