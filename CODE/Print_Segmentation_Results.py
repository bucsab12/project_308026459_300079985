import matplotlib.pyplot as plt
from Utils import *
from Models import *
import matplotlib.colors as plt_colors
import matplotlib.patches as mpatches


def print_segmentation(path_to_brains_list, model, save_path, brain_num):
    """
    This function prints out a segmentation for a selected brain from the validation brains. An image will be created for each slice comprised of the 4 modalities, the ground truth and the prediction.
    :param path_to_brains_list: Path to the list of brains used for validation
    :param model: The model to use for prediction
    :param save_path: Path where to save the output images
    :param brain_num: Index of brain we would like to segment
    :return : None
    """
    path_to_brain = path_to_brains_list[brain_num]
    X, Y = Data_Generator(path_to_brain)  # Created slices
    brain_len = Y.shape[0]
    x_padded_large = np.zeros((brain_len, 304, 304, 4))
    x_padded_small = np.zeros((brain_len, 272, 272, 4))
    x_padded_large[:, 32:-32, 32:-32, :] = X
    x_padded_small[:, 16:-16, 16:-16, :] = X
    y_slice_pred = model.predict([x_padded_large, x_padded_small], batch_size=1, use_multiprocessing=True, workers=8, verbose=2)
    for slice_n in range(brain_len):
        fig_ins, ax_ins = plt.subplots(1, 6, figsize=(20, 10))
        for m in range(4):
            ax_ins[m].set_title("Brain Modality {}".format(m))
            ax_ins[m].imshow(X[slice_n, :, :, m], vmin=0, vmax=1)
        bkg_color = 'black'
        necrotic_color = 'tomato'
        edema_color = 'lawngreen'
        en_tumor_color = 'yellow'
        cvals = [0, 1, 2, 3]
        colors = [bkg_color, necrotic_color, edema_color, en_tumor_color]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        necrotic_patch = mpatches.Patch(color=necrotic_color, label='necrotic')
        edema_patch = mpatches.Patch(color=edema_color, label='Edema')
        en_tumor_patch = mpatches.Patch(color=en_tumor_color, label='Enhanced tumor')
        fig_ins.legend(handles=[necrotic_patch, edema_patch, en_tumor_patch])
        cmap = plt_colors.LinearSegmentedColormap.from_list("", tuples)
        ax_ins[4].set_title("True label {}".format(slice_n))
        ax_ins[4].imshow(Y[slice_n], vmin=0, vmax=3, cmap=cmap, norm=norm)
        pred_labels = y_slice_pred[slice_n].argmax(axis=-1).reshape(240, 240)
        ax_ins[5].set_title("Pred label {}".format(slice_n))
        ax_ins[5].imshow(pred_labels, vmin=0, vmax=3, cmap=cmap, norm=norm)
        plt.savefig(os.path.normpath(save_path + '/Brain Modality brain num = {} slice num = {}'.format(brain_num, slice_n)))
        plt.close()


Model_Type = "Two_Path"  # "Three_Path", "DenseNet", " ResNet"  # Selects the model type
brain_number = 8  # Index of brain number used for segmentation
MODEL_LOAD_DIR = "MODELS"  # Directory from which to load the specified model
MODEL_LOAD_NAME = 'Two_Pathway.hdf5'  # Model weights file to load
MODEL_LOAD_PATH = os.path.normpath(os.path.join(MODEL_LOAD_DIR, MODEL_LOAD_NAME))
EVAL_DATA_PATH = "VALIDATION"  # Path to validation directory
brains_for_prediction_list = glob(os.path.normpath(os.path.join(EVAL_DATA_PATH, 'BraTS*')))
SAVE_PATH = "VALIDATION"  # Path where brain segmentation will be saved
save_path1 = os.path.normpath(os.path.join(SAVE_PATH, "brain number {} ".format(brain_number) + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")))
os.mkdir(save_path1)
Optimizer = Adam()
m1 = keras.models.load_model((MODEL_LOAD_PATH))
m2 = create_slice_prediction_model(optimizer=Optimizer, model=m1, model_type=Model_Type)
print_segmentation(brains_for_prediction_list, m2, save_path1, brain_num=brain_number)
