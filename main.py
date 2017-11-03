import os
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import pickle

from sklearn.cluster import DBSCAN

from minisom import MiniSom

depth = 255.0
# source_dir = "notMNIST_large/"
# source_dir = "lfw-deepfunneled/"
source_dir = "notMNIST_small/"
base_cache_dir = 'cache/'
cache_dir = base_cache_dir + source_dir
output_dir = 'dataset/'

# must add to 1
part = 1
train = 0.4 * part
test = 0.4 * part
valid = 0.1 * part

numlabels = 0

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

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
            img = cv2.imread(source_dir + "/" + label + "/" + image, 0)
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
            plt.subplot(numlabels + 1, 3, grid_item)
            grid_item = grid_item + 1
            plt.axis("off")
            plt.imshow(pickle_data[random.randint(0, len(pickle_data) - 1)])
    plt.title('first 3 letters')
    plt.show()


# letter_pickle_preview()

def pickle_join_labels():
    if (os.path.exists(base_cache_dir + source_dir.replace('/', '') + '_labels.pickle')):
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

        print('dumping')
    pickle.dump(all_data, open(base_cache_dir + source_dir.replace('/', '') + '_data.pickle', 'wb'))
    pickle.dump(all_labels, open(base_cache_dir + source_dir.replace('/', '') + '_labels.pickle', 'wb'))


pickle_join_labels()


# PICKE ALL TO ONE
def pickle_letters_to_dataset():
    if (os.path.exists(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle')):
        return

    print('reading')
    all_data = pickle.load(open(base_cache_dir + source_dir.replace('/', '') + '_data.pickle', 'rb'))
    all_labels = pickle.load(open(base_cache_dir + source_dir.replace('/', '') + '_labels.pickle', 'rb'))
    print('read')

    data_size = len(all_labels)
    permutation = np.random.permutation(data_size)

    print('randomizing: ')
    all_data = np.asarray(all_data)
    print('d1')
    all_data = all_data[permutation]
    print('d2')
    all_labels = np.asarray(all_labels)
    all_labels = all_labels[permutation]

    print('splitting')
    train_data = all_data[:int(train * data_size), :, :]
    train_labels = all_labels[:int(train * data_size)]
    test_data = all_data[(int(train * data_size) + 1):(int((train + test) * data_size)), :, :]
    test_labels = all_data[(int(train * data_size) + 1):(int((train + test) * data_size))]
    valid_data = all_data[(int((train + test) * data_size) + 1): (int((train + test + valid) * data_size)), :, :]
    valid_labels = all_data[(int(((train + test) * data_size)) + 1):(int((train + test + valid) * data_size))]

    all_pickle = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels,
    }

    print('dumping result to file.')
    pickle.dump(all_pickle, open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'wb'))


pickle_letters_to_dataset()


def read_data():
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'rb'))
    print('train data: ' + str(len(data['train_labels'])))
    print('test data: ' + str(len(data['test_labels'])))
    print('valid data: ' + str(len(data['valid_labels'])))

    grid_item = 1
    for i in range(0, 10):
        plt.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        plt.axis("off")
        plt.imshow(data['train_data'][i])
    plt.title('train')

    for i in range(0, 10):
        plt.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        plt.axis("off")
        plt.imshow(data['test_data'][i])
    plt.title('test')

    for i in range(0, 10):
        plt.subplot(3, 10, grid_item)
        grid_item = grid_item + 1
        plt.axis("off")
        plt.imshow(data['valid_data'][i])
    plt.title('valid')
    plt.show()


# read_data()

# ZADANIE 2

diff = 0.0001


def intersect(images1, images2, labels2):
    keep2 = list(range(0, len(images2)))
    print('going: ' + str(len(images1)) + ' X ' + str(len(images2)))
    for i in range(0, len(images1)):
        if (i % 100 == 0):
            print('.', end='')
        for j in keep2:
            # if ((images1[i] == images2[j]).all()):
            if (abs(np.mean(abs(images1[i] - images2[j]))) < diff):
                keep2.remove(j)

    indices = np.asarray(keep2)
    images2 = images2[indices]
    labels2 = labels2[indices]
    return images2, labels2


def basic_intersect():
    print('BASIC INTERSECT')
    if (os.path.exists(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle')):
        print('skipping.')
        return

    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'rb'))
    print('train data: ' + str(len(data['train_labels'])))
    print('test data: ' + str(len(data['test_labels'])))
    print('valid data: ' + str(len(data['valid_labels'])))

    data['test_data'], data['test_labels'] = intersect(data['train_data'], data['test_data'], data['test_labels'])
    print()
    print('test data: ' + str(len(data['test_labels'])))

    data['valid_data'], data['valid_labels'] = intersect(data['test_data'], data['valid_data'], data['valid_labels'])
    print()
    print('test data: ' + str(len(data['valid_labels'])))

    data['train_data'], data['train_labels'] = intersect(data['valid_data'], data['train_data'], data['train_labels'])
    print()
    print('train data: ' + str(len(data['train_labels'])))
    pickle.dump(data, open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'wb'))


basic_intersect()


def find_closest(center, data, labels):
    min = 2
    index = 0
    for i in range(0, len(data)):
        temp = abs(np.mean(center - data[i]))
        if (temp < min):
            min = temp
            index = i
    return data[index], labels[index]


def k_means():
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'rb'))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    num_labels = 10

    in_data = data['train_data']
    in_labels = data['train_labels']
    img_dims = len(in_data[0])
    in_data = np.float32(in_data)
    in_data = np.reshape(in_data, (len(in_data), img_dims * img_dims))

    compactness, label, centers = cv2.kmeans(in_data, num_labels, None, criteria, 10, flags)

    i = 1
    for img in centers:
        plt.subplot(num_labels, 2, i)
        closest, label = find_closest(img, in_data, in_labels)
        img = np.reshape(img, (img_dims, img_dims))
        closest = np.reshape(closest, (img_dims, img_dims))
        plt.imshow(closest)
        i = i + 1

        plt.subplot(num_labels, 2, i)
        plt.imshow(img)
        i = i + 1

    plt.show()

# k_means()

def make_average(labels, data):
    labels_data = []
    label_set = set(labels)
    print(len(label_set), label_set)
    for label in label_set:
        if label < 0:
            continue

        cluster = np.where(label == labels)[0]
        if (len(cluster) == 0):
            continue

        cluster_data = data[cluster, :]
        cluster_data = np.average(cluster_data, 0)
        labels_data.append(cluster_data)

    return labels_data


def my_dbscan():
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'rb'))
    in_data = data['train_data']
    in_labels = data['train_labels']

    img_dims = len(in_data[0])
    in_data = np.reshape(in_data, (len(in_data), img_dims * img_dims))

    db = DBSCAN(eps=7.4, min_samples=60 * part, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30,
                p=None, n_jobs=1).fit(in_data)
    # plt.hist(db.labels_)
    # plt.show()

    centers = make_average(db.labels_, in_data)
    num_labels = len(centers)
    print(num_labels)
    i = 1
    for img in centers:
        plt.subplot(num_labels, 2, i)
        closest, label = find_closest(img, in_data, in_labels)
        img = np.reshape(img, (img_dims, img_dims))
        closest = np.reshape(closest, (img_dims, img_dims))
        plt.imshow(closest)
        i = i + 1

        plt.subplot(num_labels, 2, i)
        plt.imshow(img)
        i = i + 1

    plt.show()


# my_dbscan()


def som():
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'rb'))
    train_data = data['train_data']
    train_labels = data['train_labels']
    test_data = data['test_data']
    test_labels = data['test_labels']
    img_dims = len(train_data[0])
    som = MiniSom(2, 5, img_dims * img_dims, sigma=0.3, learning_rate=0.5)

    print("Training...")
    train_data = np.reshape(train_data, [len(train_data), img_dims * img_dims])
    som.train_batch(train_data, len(train_data))
    som.train_random(train_data, 100)
    print("...ready!")

    test_data = np.reshape(test_data, [len(test_data), img_dims * img_dims])
    labels = []
    for img in test_data:
        x = som.winner(img)
        # print(x)
        labels.append(x[0] * 5 + x[1])

    centers = make_average(labels, test_data)
    num_labels = len(centers)
    i = 1
    for img in centers:
        plt.subplot(num_labels, 2, i)
        closest, label = find_closest(img, test_data, test_labels)
        img = np.reshape(img, (img_dims, img_dims))
        closest = np.reshape(closest, (img_dims, img_dims))
        plt.imshow(closest)
        i = i + 1

        plt.subplot(num_labels, 2, i)
        plt.imshow(img)
        i = i + 1

    plt.show()

som()
