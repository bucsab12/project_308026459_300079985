# Imports
from keras.optimizers import SGD
from Utils import *
from segmentation_models.losses import cce_jaccard_loss


# Path Declaration
LOGS_SAVE_PATH = "LOGS"               # Path to log directory
MODEL_SAVE_PATH = "MODELS"            # Path where to save model files
TRAIN_DATA_PATH = "TRAIN"             # Path to training data
EVAL_DATA_PATH = "VALIDATION"         # Path to validation data
MODEL_LOAD_DIR = "MODELS"             # Path to directory with the best weights for each model
MODEL_LOAD_NAME = 'Two_Pathway.hdf5'  # Name of best weights model to be loaded
MODEL_LOAD_PATH = os.path.normpath(os.path.join(MODEL_LOAD_DIR, MODEL_LOAD_NAME))
brains_for_train_list = glob(os.path.join(TRAIN_DATA_PATH, "BraTS19_2013*"))
brains_for_prediction_list = glob(os.path.normpath(os.path.join(EVAL_DATA_PATH, 'BraTS*')))

# Hyper Parameters
TRAIN_FROM_SCRATCH = True                                      # Determines whether to start training the model from scratch if True or load an existing model if False
TRAIN_MODEL = True                                             # Determines if we are in training mode in True or in prediction mode if False
Model_Type = "Two_Path"                                        # "Three_Path", "DenseNet", " ResNet"        # Selects the model type
Num_of_Epochs = 80                                             # Number of epochs used for training. An epoch is defined as one pass over the 30 brains in the BraTS 2013 dataset.
Batch_Size = 64                                                # Batch size
Use_Cyclic_LR = True                                           # Determines whether cyclic learning rate will be used
LR_min = 0.000005                                              # Minimal learning rate when using cyclic learning rate.
LR_max = 0.01                                                  # Maximal learning rate when using cyclic learning rate.
LR_start = 0.0005                                              # Initial learning rate value that will be used when using standard learning rate decay (Use_Cyclic_LR=False)
Decay = 0.1                                                    # Decay value to be used when running with standard learning rate decay
Optimizer = SGD(LR_start, momentum=0.9, clipvalue=0.5)         # Selected optimizer
Num_of_Class_Patches = 200                                     # Number of random patches that will be generated from each class, per brain, during training
Num_of_Class_Patches_validation = 200                          # Number of random patches that will be generated from each class, per brain, during validation at each fit command. For average dice calculation, the entire brains will be used and not just select patches.
Repetitions_Per_Brain = 1                                      # Determines the number of times the training process will be repeated for each brain
Fit_Epochs = 1                                                 # Determines the number of epochs for the training process on each mini batch of brains
Brain_Mini_Batch_Size = 6                                      # Determines the number of brains in each mini batch


if TRAIN_MODEL:
    if TRAIN_FROM_SCRATCH:
        # Model generation
        if Model_Type == "Two_Path":
            m1 = input_cascade_two_path((65, 65, 4), (33, 33, 4))
        if Model_Type == "Three_Path":
            m1 = input_cascade_three_path((65, 65, 4), (33, 33, 4))
        if Model_Type == "DenseNet":
            m1 = input_cascade_densenet((65, 65, 4), (33, 33, 4))
        if Model_Type == "ResNet":
            m1 = input_cascade_resnet((65, 65, 4), (33, 33, 4))
        m1.summary()
        # Compile the model
        m1.compile(optimizer=Optimizer, loss=cce_jaccard_loss, metrics=["categorical_accuracy"])
    else:
        # Load Model
        m1 = keras.models.load_model((MODEL_LOAD_PATH))
        m1._name = 'Loaded_Model_{} from time'.format(str(Model_Type)) + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")

    # Model Checkpoint Path creation
    localtime = str(datetime.utcnow() + timedelta(hours=2)).split(".")[0]
    Local_Model_PATH = os.path.normpath(os.path.join(MODEL_SAVE_PATH, str(m1._name)))
    Local_Log_PATH = os.path.normpath(os.path.join(LOGS_SAVE_PATH, str(m1._name)))
    os.mkdir(Local_Model_PATH)
    os.mkdir(Local_Log_PATH)
    plot_model(m1, to_file=os.path.normpath(os.path.join(Local_Log_PATH, 'model_plot.png')), show_shapes=True, show_layer_names=True)

    # Training
    iterations = 1
    best_dice = 0
    brains_patches_list = create_list_of_patches(brains_for_train_list)
    brains_patches_list_validation = create_list_of_patches(brains_for_prediction_list)
    iterations_per_epoch = len(brains_patches_list) // Brain_Mini_Batch_Size
    print("Iteration per Epoch: {}\n".format(iterations_per_epoch))
    for ep in range(Num_of_Epochs):
        print("Epoch number: {} out of {}\n".format(ep, Num_of_Epochs))
        rand_brain_order = np.arange(len(brains_patches_list))
        rand_brain_order_validation = np.arange(len(brains_patches_list_validation))
        random.shuffle(rand_brain_order)
        random.shuffle(rand_brain_order_validation)
        brain_number = 0
        start_idx = 0
        end_idx = start_idx + Brain_Mini_Batch_Size
        brains_for_train_list_shuffled = list(np.array(brains_for_train_list)[rand_brain_order])
        # brains_for_train_list_shuffled = brains_for_train_list
        while end_idx <= len(brains_for_train_list_shuffled):
            print("\niteration number = {}\n".format(iterations))
            X_large_patch_brain_mb, X_small_patch_brain_mb, Y_slice_brain_mb, _, _ = \
                batch_generator_main(brain_number, start_idx, end_idx, brains_for_train_list_shuffled, Repetitions_Per_Brain, brains_patches_list, Num_of_Class_Patches, Brain_Mini_Batch_Size, rand_brain_order)
            X_large_patch_brain_mb_validation, X_small_patch_brain_mb_validation, Y_slice_brain_mb_validation, _, _ = \
                batch_generator_main(0, 0, 6, brains_for_prediction_list, 1, brains_patches_list_validation, Num_of_Class_Patches_validation, 0, rand_brain_order_validation)

            m1.fit([X_large_patch_brain_mb, X_small_patch_brain_mb], Y_slice_brain_mb,
                   validation_data=([X_large_patch_brain_mb_validation, X_small_patch_brain_mb_validation], Y_slice_brain_mb_validation),
                   epochs=Fit_Epochs, batch_size=Batch_Size,
                   verbose=2, use_multiprocessing=True, workers=8)

            if ((iterations % (iterations_per_epoch//2)) == 0) and (iterations != 0):
                m2 = create_slice_prediction_model(optimizer=Optimizer, model=m1, model_type=Model_Type)
                logs = predict_brain_full(brains_for_prediction_list, m2)
                average_dice_cost = (logs[1] + logs[2] + logs[3]) / 3
                print('average DICE cost for classes, 1, 2, 4: {}\n'.format(average_dice_cost))
                if average_dice_cost > best_dice:
                    m1.save(os.path.normpath(Local_Model_PATH + "/weights.iterations=" + str(iterations) + "_average_dice_cost=" + str(average_dice_cost) + ".hdf5"))
                    print('DICE score improved from {} to {}\n'.format(best_dice, average_dice_cost))
                    best_dice = average_dice_cost
            iterations += 1
            # Learning Rate Change
            if Use_Cyclic_LR:
                temp_lr = K.get_value(Optimizer.lr)
                lr = cyclic_lr(iterations, base_lr=LR_min, max_lr=LR_max, step_size=float(iterations_per_epoch / 2), gamma=0.992)
                print('\nChanged learning rate from {} to {}\n'.format(temp_lr, lr))
                K.set_value(Optimizer.lr, lr)
            elif (iterations % 200 == 0) and (iterations != 0):
                temp_lr = K.get_value(Optimizer.lr)
                lr = temp_lr * 0.1
                print('\nChanged learning rate from {} to {}\n'.format(temp_lr, lr))
                K.set_value(Optimizer.lr, lr)
else:
    m2 = create_slice_prediction_model(optimizer=Optimizer, model_load_path=MODEL_LOAD_PATH, load_model=True, model_type=Model_Type)
    brains_for_prediction_list = glob(os.path.normpath(os.path.join(EVAL_DATA_PATH, 'BraTS*')))
    logs = predict_brain_full(brains_for_prediction_list, m2)
