from glob import glob
import SimpleITK as sitk
import os
import numpy as np
import random
from keras.utils import to_categorical
from Models import *
from imgaug import augmenters as iaa


def augment():
    """
    Function used for augmentation of patches. Defines the pipeline of augmentation that will be used on each patch.
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # vertically flip 50% of the images
        iaa.Flipud(0.5),  # horizontally flip 50% of the images
        iaa.Sometimes(0.5, iaa.Rot90((1, 3)))  # Rotate by 90, 180 or 270 degrees 50% of the images
    ])
    return seq


def batch_generator(pat_folders_shuffled, start_idx, end_idx, brain_number, repetitions_per_brain, brains_patches_list, num_of_patches_for_class, brain_mini_batch_size, rand_brain_order):
    """
    The function loads a specified number of full brain images from a random list of brains
    and creates a mini batch of large patches, small patches and one-hot labels with respect to the centers of the patches.
    The patches are drawn with equal probability from each class from all the brains
    :param pat_folders_shuffled:a shuffled list of paths to brain folders
    :param start_idx:idx from which to start loading brain
    :param end_idx:the idx of the last brain
    :param brain_number: index for keeping track on the current brain number
    :param repetitions_per_brain: sets how many times we extract random patches from the same brain before loading the next one.
    :param brains_patches_list: a list containing all the patches center pixels in the dataset ,shape = (brain, class, (slice1, x1, y1)...(sliceK, xK, yK))
    :param num_of_patches_for_class: number of extracted patches per class
    :param brain_mini_batch_size: from how many brains to extract patches
    :return: indexes for outer function handling (brain_number, start_idx, end_idx)
    y_patches_brain_mb_list: list containing labels of patches, shape (num of patches, 1, 1, num of classes)
    x_large_patches_brain_mb_list: list containing large patches, shape (num of patches, 65, 65, num of channels)
    x_small_patches_brain_mb_list: list containing small patches, shape (num of patches, 33, 33, num of channels)
    """
    x_large_patches_brain_mb_list = []
    x_small_patches_brain_mb_list = []
    y_patches_brain_mb_list = []
    wight_mb_list = []
    print("\nLoading random brains")
    for brain in pat_folders_shuffled[start_idx:end_idx]:
        X, Y = Data_Generator(brain)
        print("current brain: {}".format(os.path.basename(brain)))
        for i in range(repetitions_per_brain):
            D, num_of_classes = create_equiprobable(X, Y, rand_brain_order[brain_number], brains_patches_list, num_of_patches_for_class)
            num_of_patches_total = D[0].shape[0]
            print("num of patches = " + str(num_of_patches_total))
            x_large_patches_brain_mb_list.append(D[0])
            x_small_patches_brain_mb_list.append(D[1])
            y_patches_brain_mb_list.append(D[2])
        brain_number += 1
    start_idx = end_idx
    end_idx = end_idx + brain_mini_batch_size
    return brain_number, start_idx, end_idx, y_patches_brain_mb_list, x_large_patches_brain_mb_list, x_small_patches_brain_mb_list, wight_mb_list


def batch_generator_main(brain_number, start_idx, end_idx, pat_folders_shuffled, repetitions_per_brain, brains_patches_list, num_of_class_patches, brain_mini_batch_size, rand_brain_order):
    brain_number, start_idx, end_idx, y_patches_brain_mb_list, x_large_patch_brain_mb_list, x_small_patch_brain_mb_list, wight_arr_mb_list = \
        batch_generator(pat_folders_shuffled, start_idx, end_idx, brain_number, repetitions_per_brain, brains_patches_list, num_of_class_patches, brain_mini_batch_size, rand_brain_order)
    num_of_patches_mb = np.sum([x.shape[0] for x in x_small_patch_brain_mb_list])
    print("\nnumber of patches for mini batch = {} ".format(num_of_patches_mb))
    rand_patches_order = np.arange(num_of_patches_mb)
    random.shuffle(rand_patches_order)
    x_large_patch_brain_mb = np.concatenate(x_large_patch_brain_mb_list)[rand_patches_order]
    x_small_patch_brain_mb = np.concatenate(x_small_patch_brain_mb_list)[rand_patches_order]
    y_slice_brain_mb = np.concatenate(y_patches_brain_mb_list)[rand_patches_order]
    return x_large_patch_brain_mb, x_small_patch_brain_mb, y_slice_brain_mb, start_idx, end_idx


def Data_Generator(folder):
    """
    function for loading nii.gz files arranging them in numpy arrays
    preforms pre-processing:
    1. clipping of 0.025 highest pixel values
    2. normalize between 0 and 1
    :param folder: folder path containing the nii.gz files
    :return:
        _data: numpy array shape : (slices, height, width, channels)
        _y_labels: numpy array shape : (slices, height, width, classes)
    """
    files = glob(os.path.join(folder, "*.nii.gz"))
    arr = []
    y_labels = []
    for i in range(len(files)):
        if not os.path.basename(files[i]).endswith('seg.nii.gz'):
            img = sitk.ReadImage(files[i])
            arr.append(sitk.GetArrayFromImage(img))
        else:
            img = sitk.ReadImage(files[i])
            y_labels = sitk.GetArrayFromImage(img)
            y_labels[y_labels == 4] = 3
    data = np.zeros((y_labels.shape[0], y_labels.shape[1], y_labels.shape[2], 4))
    for i in range(y_labels.shape[0]):
        data[i, :, :, 0] = arr[0][i, :, :]
        data[i, :, :, 1] = arr[1][i, :, :]
        data[i, :, :, 2] = arr[2][i, :, :]
        data[i, :, :, 3] = arr[3][i, :, :]
    _data = np.zeros((data.shape))
    for idx in range(4):
        channel_slices = data[:, :, :, idx]
        b, t = np.percentile(channel_slices, (0, 99.975))
        channel_slices_clip = np.clip(channel_slices, b, t)
        _data[:, :, :, idx] = (channel_slices_clip - b) / (t - b)
    return _data, y_labels


def create_equiprobable(x, y, brain_num, brains_patches_list, num_of_patches):
    """
    the function extracts patches from a given brain image.
    the function uses a list which contains all the pixel centers in which a non background/healthy class is present
    and checks if it valid for extraction of a large patch from it.
    it extracts from every pixel a small patch of size (33, 33, 4) an a large patch (65, 65, 4) and the corresponding label.
    the label is transformed to a one-hot representation.
    :param x: numpy array containing the brain, shape : (slices, height, width, channels)
    :param y: numpy array containing the labels of the brain, shape : (slices, height, width, classes)
    :param brain_num: number of brain from which to extract patches
    :param brains_patches_list: a list containing all the patches center pixels in the dataset ,shape = (brain, class, (slice1, x1, y1)...(sliceK, xK, yK))
    :param num_of_patches: number of extracted patches per class
    :return:
        d: [(x_patches_large),(x_patches_small),(y_patches)] data containing all the patches large, small and the labels
            y_patches: list containing labels of patches, shape (num of patches, 1, 1, num of classes)
            x_patches_large: list containing large patches, shape (num of patches, 65, 65, num of channels)
            x_patches_small: list containing small patches, shape (num of patches, 33, 33, num of channels)
        num_of_classes: num of classes that was found in the brain (LGG brains don't usually have class number 4)
    """
    brain_patches_list = brains_patches_list[brain_num]
    x_patches_large = []
    y_patches = []
    input_dim = 65
    if len(brain_patches_list[3][0]) == 0:
        num_of_classes = 3
    else:
        num_of_classes = 4
    for i in range(num_of_classes):
        rand_idx = random.choices(np.arange(len(brain_patches_list[i][0])), k=num_of_patches)
        patches_rand_coordinates = [brain_patches_list[i][0][x] for x in rand_idx]
        for co in patches_rand_coordinates:
            if (co[1] + input_dim//2 + 1 > 240) or (co[1] - input_dim//2 < 0) or (co[2] + input_dim//2 + 1 > 240) or (co[2] - input_dim//2 < 0):
                continue
            else:
                x_patches_large.append(x[co[0], co[1] - int(input_dim / 2):co[1] + int(input_dim / 2) + 1, co[2] - int(input_dim / 2):co[2] + int(input_dim / 2) + 1, :])
                y_patches.append(y[co[0], co[1], co[2]])
    augmenter = augment()
    _x_patches_large = augmenter(images=x_patches_large)
    x_patches_small = np.array(_x_patches_large)[:, 16:-16, 16:-16, :]
    num_of_patches_total = np.asarray(y_patches).shape[0]
    y_patches_per_brain = np.zeros((num_of_patches_total, 1, 1, 4))
    for j in range(num_of_patches_total):
        y_patches_per_brain[j, :, :, y_patches[j]] = 1
    d = [np.asarray(_x_patches_large), np.asarray(x_patches_small), np.asarray(y_patches_per_brain)]
    return d, num_of_classes


def create_list_of_patches(list_of_brain_folders):
    """
    creates a list which contains all the pixel centers in which a non background/healthy class is present
    :param list_of_brain_folders: list containing all the paths to the brain folders
    :return: brains_list: list in the form of (brain, class, (slice1, x1, y1)...(sliceK, xK, yK))
    """
    brains_list = []
    for num_brain, folder in enumerate(sorted(list_of_brain_folders)):
        slices_list = [[], [], [], []]
        x, y = Data_Generator(folder)
        slices_list[0].append(np.argwhere((x[:, :, :, 0] != 0) & (y == 0)))
        slices_list[1].append(np.argwhere(y == 1))
        slices_list[2].append(np.argwhere(y == 2))
        slices_list[3].append(np.argwhere(y == 3))
        brains_list.append(slices_list)
    return brains_list


def create_slice_prediction_model(optimizer, model_type="Two_Path", model_load_path='', load_model=False, model=''):
    """
    creates a new model with a new input shape for inference of full size images and not patches.
    it copies the weights from the trained model on the patches to a new model which excepts full image
    :param optimizer: optimizer for compile
    :param model_type: choose between TwoPath Way model if True or other model if False
    :param model_load_path: path to model checkpoint
    :param load_model: choose whether to load a trained model or evaluate from current model
    :param model: model parameter if load model is not used
    :return:model for full image inference
    """
    if load_model:
        model1 = keras.models.load_model(model_load_path)
    else:
        model1 = model

    if model_type == "Two_Path":
        model2 = input_cascade_two_path((None, None, 4), (None, None, 4))
    if model_type == "Three_Path":
        model2 = input_cascade_three_path((None, None, 4), (None, None, 4))
    if model_type == "DenseNet":
        model2 = input_cascade_densenet((None, None, 4), (None, None, 4))
    if model_type == "RenseNet":
        model2 = input_cascade_resnet((None, None, 4), (None, None, 4))

    model2.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    for layer in range(len(model2.layers)):
        model2.layers[layer].set_weights(model1.layers[layer].get_weights())
    return model2


def predict_brain_full(path_to_brains_list, model, mode='train'):
    """
    evaluates model with dice score for each class on the evaluation dataset and an average dice score on the 3 non-background classes
    evaluation is preformed on a full brain image using the inference model
    :param path_to_brains_list: list of paths to the evaluation brains
    :param model: inference model for prediction
    :return: logs: list containing dice for each class and an average on the 3 non-background classes
    """
    brain_pred_labels_list = []
    brain_true_labels_list = []
    print('performing validation')
    num_of_validation_brains = len(path_to_brains_list)
    for brain_num, path_to_brain in enumerate(path_to_brains_list):
        X, Y = Data_Generator(path_to_brain)  # Created slices
        brain_len = Y.shape[0]
        print('\nvalidation brain number {}/{}: {}'.format(brain_num+1, num_of_validation_brains, os.path.basename(path_to_brain)))
        if mode == 'train':
            min_row19 = 29
            max_row19 = 220
            min_col19 = 40
            max_col19 = 196
            new_height = max_row19 - min_row19
            new_width  = max_col19 - min_col19
            x_padded_large = np.zeros((brain_len, new_height + 64, new_width + 64, 4))
            x_padded_small = np.zeros((brain_len, new_height + 32, new_width + 32, 4))
            x_padded_large[:, 32:-32, 32:-32, :] = X[:, min_row19: max_row19, min_col19:max_col19, :]
            x_padded_small[:, 16:-16, 16:-16, :] = X[:, min_row19: max_row19, min_col19:max_col19, :]
            label_1hot = to_categorical(Y[:, min_row19: max_row19, min_col19:max_col19], 4)
        else:
            x_padded_large = np.zeros((brain_len, 304, 304, 4))
            x_padded_small = np.zeros((brain_len, 272, 272, 4))
            x_padded_large[:, 32:-32, 32:-32, :] = X
            x_padded_small[:, 16:-16, 16:-16, :] = X
            label_1hot = to_categorical(Y, 4)
        # D = [x_padded_large, x_padded_small, Y]
        y_pred = model.predict([x_padded_large, x_padded_small], batch_size=1, use_multiprocessing=True, workers=8, verbose=2)
        brain_pred_labels_list.append(y_pred)
        brain_true_labels_list.append(label_1hot)
    brain_pred_labels = np.concatenate(brain_pred_labels_list)
    brain_true_labels = np.concatenate(brain_true_labels_list)
    np.savez_compressed(os.path.normpath(os.path.join(os.path.dirname(path_to_brain), 'val_results')), a=brain_true_labels, b=brain_pred_labels)  # Save the ground truth and predicted labels for a given set of validation brains
    logs = numpy_calc_dice_2(brain_true_labels, brain_pred_labels)
    print('\n')
    for idx, metric in enumerate(logs):
        print("DICE score for class {} over brain {} is: {}".format(idx, os.path.basename(path_to_brain), metric))
    return logs


def cyclic_lr(iterations, base_lr=0.001, max_lr=0.006, step_size=15., gamma=0.992, mode='triangular'):
    """
    computes the learning rate with the exponential cyclic learning rate method
    :param iterations:iteration counter
    :param base_lr:minimal learning rate
    :param max_lr:maximal learning rate
    :param step_size: how many steps to change lr
    :param gamma:the basis constant for the exponential lr
    :return:learning rate
    """
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)
    if mode == 'triangular':
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    if mode == 'annealing':
        lr = base_lr + (max_lr - base_lr) * (1 + np.cos(np.pi * iterations / step_size)) / 2
    if mode == 'decreasing exp':
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** iterations
    return lr


def numpy_calc_dice_2(y_true, y_pred):
    """
    :param y_true: Ground truth mask for pixels in given data set
    :param y_true: Predicted mask for pixels in given data set
    :return : Dice scores for each of the classes in the data set
    """
    epsilon = np.finfo(float).eps
    num_tot = 2.0 * np.sum(y_true * y_pred, (0, 1, 2)) + epsilon
    den_tot = np.sum(y_true, (0, 1, 2)) + np.sum(y_pred, (0, 1, 2)) + epsilon
    return num_tot / den_tot
