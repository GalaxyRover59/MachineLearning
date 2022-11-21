import os
import random
import shutil
import numpy as np

DATASET_DIR = 'new_17_flowers'  # 数据集路径
NEW_DIR = 'data'  # 数据切分后存放路径
num_test = 0.2  # 测试集占比


# 打乱所有种类数据，并分割训练集和测试集
def shuffle_all_files(dataset_dir, new_dir, num_test):
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    train_dir = os.path.join(new_dir, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(new_dir, 'test')
    os.mkdir(test_dir)

    directories = []
    train_directories = []
    test_directories = []
    class_names = []

    # 原数据集中所有类别分别在train和test文件夹中创建相应子文件夹
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            train_directories.append(train_path)
            os.mkdir(train_path)
            test_directories.append(test_path)
            os.mkdir(test_path)
            class_names.append(filename)
    print('类别列表：', class_names)

    for i in range(len(directories)):
        photo_filenames = []
        train_photo_filenames = []
        test_photo_filenames = []
        # 将某种类文件夹中的文件路径加入numpy数组，并为train和test构建相应的数组
        for filename in os.listdir(directories[i]):
            path = os.path.join(directories[i], filename)
            train_path = os.path.join(train_directories[i], filename)
            test_path = os.path.join(test_directories[i], filename)
            photo_filenames.append(path)
            train_photo_filenames.append(train_path)
            test_photo_filenames.append(test_path)
        photo_filenames = np.array(photo_filenames)
        train_photo_filenames = np.array(train_photo_filenames)
        test_photo_filenames = np.array(test_photo_filenames)

        index = [j for j in range(len(photo_filenames))]
        random.shuffle(index)  # 将索引打乱
        photo_filenames = photo_filenames[index]
        train_photo_filenames = train_photo_filenames[index]
        test_photo_filenames = test_photo_filenames[index]

        test_sample_index = int((1 - num_test) * float(len(photo_filenames)))
        for j in range(test_sample_index):
            shutil.copyfile(photo_filenames[j], train_photo_filenames[j])
        for j in range(test_sample_index, len(photo_filenames)):
            shutil.copyfile(photo_filenames[j], test_photo_filenames[j])

shuffle_all_files(DATASET_DIR, NEW_DIR, num_test)
