from keras import layers
from keras.layers import Multiply
from datetime import datetime
from datetime import timedelta
from DenseNet_Model import *
from ResNet_Model import *
from keras.regularizers import l1_l2, l2


def three_path(X_input):
    """
    Three pathway block.
    :param X_input: Input to the block
    :return: Output of the block
    """
    # Small Path
    X0 = Conv2D(64, (5, 5), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X0 = LeakyReLU(alpha=0.3)(X0)
    X0 = BatchNormalization()(X0)
    X0 = Conv2D(64, (5, 5), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X0)
    X0 = LeakyReLU(alpha=0.3)(X0)
    X0 = BatchNormalization()(X0)
    X0 = Conv2D(64, (5, 5), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X0)
    X0 = LeakyReLU(alpha=0.3)(X0)
    X0 = BatchNormalization()(X0)

    # Local path
    X = Conv2D(64, (7, 7), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)
    X1 = Conv2D(64, (7, 7), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X1 = LeakyReLU(alpha=0.3)(X1)
    X1 = BatchNormalization()(X1)
    X = layers.Maximum()([X, X1])
    X = Conv2D(64, (4, 4), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)
    X3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X)
    X3 = LeakyReLU(alpha=0.3)(X3)
    X3 = BatchNormalization()(X3)
    X31 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X)
    X31 = LeakyReLU(alpha=0.3)(X31)
    X31 = BatchNormalization()(X31)
    X = layers.Maximum()([X3, X31])
    X = Conv2D(64, (2, 2), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)

    # Global path
    X2 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X2 = LeakyReLU(alpha=0.3)(X2)
    X2 = BatchNormalization()(X2)
    # X21 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid', activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X21 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(0.01, 0.01))(X_input)
    X21 = LeakyReLU(alpha=0.3)(X21)
    X21 = BatchNormalization()(X21)
    X2 = layers.Maximum()([X2, X21])

    # Merging the two paths
    X = Concatenate()([X2, X, X0])
    return X


def input_cascade_three_path(input_shape1, input_shape2):
    """
    Cascaded Three-Pathway architecture
    :param input_shape1: Large patch
    :param input_shape2: small patch
    :return: Cascaded Three-Pathway model
    """
    X1_input = Input(input_shape1)
    # 1st three-path of cascade
    X1 = three_path(X1_input)
    X1 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid')(X1)
    X1 = LeakyReLU(alpha=0.3)(X1)
    X1 = BatchNormalization()(X1)

    X2_input = Input(input_shape2)
    # Concatenating the output of 1st to input of 2nd
    X2_input1 = Concatenate()([X1, X2_input])
    X2 = three_path(X2_input1)
    X2 = Conv2D(4, (21, 21), strides=(1, 1), padding='valid')(X2)
    X2 = Activation('softmax')(X2)

    model = Model(inputs=[X1_input, X2_input], outputs=X2)
    model._name = 'input_cascade_three_path_way' + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")
    return model


def two_path(X_input):
    """
    Two pathway block.
    :param X_input: Input to the block
    :return: Output of the block
    """
    # Local path
    X = Conv2D(64, (7, 7), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X_input)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)
    X1 = Conv2D(64, (7, 7), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X_input)
    X1 = LeakyReLU(alpha=0.3)(X1)
    X1 = BatchNormalization()(X1)
    X = layers.Maximum()([X, X1])
    X = Conv2D(64, (4, 4), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)
    X3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X)
    X3 = LeakyReLU(alpha=0.3)(X3)
    X3 = BatchNormalization()(X3)
    X31 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X)
    X31 = LeakyReLU(alpha=0.3)(X31)
    X31 = BatchNormalization()(X31)
    X = layers.Maximum()([X3, X31])
    X = Conv2D(64, (2, 2), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = BatchNormalization()(X)

    # Global path
    X2 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X_input)
    X2 = LeakyReLU(alpha=0.3)(X2)
    X2 = BatchNormalization()(X2)
    X21 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X_input)
    X21 = LeakyReLU(alpha=0.3)(X21)
    X21 = BatchNormalization()(X21)
    X2 = layers.Maximum()([X2, X21])

    # Merging the two paths
    X = Concatenate()([X2, X])
    return X


def input_cascade_two_path(input_shape1, input_shape2):
    """
    Cascaded Two-Pathway architecture
    :param input_shape1: Large patch
    :param input_shape2: small patch
    :return: Cascaded Two-Pathway model
    """
    X1_input = Input(input_shape1)
    # 1st two-path of cascade
    X1 = two_path(X1_input)
    X1 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X1)
    X1 = LeakyReLU(alpha=0.3)(X1)
    X1 = BatchNormalization()(X1)
    X2_input = Input(input_shape2)
    X2_input1 = Concatenate()([X1, X2_input])

    # 2st two-path of cascade
    X2 = two_path(X2_input1)
    X2 = Conv2D(4, (21, 21), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.0005))(X2)
    X2 = Activation('softmax')(X2)

    model = Model(inputs=[X1_input, X2_input], outputs=X2)
    model._name = 'input_cascade_two_path_way' + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")
    return model


def input_cascade_densenet(input_shape1, input_shape2):
    """
    Cascaded Densenet architecture
    :param input_shape1: Large patch
    :param input_shape2: small patch
    :return: Cascaded Densenet model
    """
    X1_input = Input(input_shape1)
    # 1st two-path of cascade
    X1 = DenseNet(X1_input, depth=7, nb_dense_block=17, growth_rate=4, nb_filter=8)
    X1 = BatchNormalization()(X1)
    X2_input = Input(input_shape2)

    # Concatenating the output of 1st to input of 2nd
    X2_input1 = Concatenate()([X1, X2_input])
    X2 = DenseNet(X2_input1, depth=7, nb_dense_block=17, growth_rate=4, nb_filter=8)

    # Fully convolutional softmax classification
    X2 = BatchNormalization()(X2)
    X2 = Activation('softmax')(X2)

    model = Model(inputs=[X1_input, X2_input], outputs=X2)
    model._name = 'input_cascade_densenet' + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")
    return model


def input_cascade_resnet(input_shape1, input_shape2):
    """
    Cascaded Resnet architecture
    :param input_shape1: Large patch
    :param input_shape2: small patch
    :return: Cascaded Resnet model
    """
    X1_input = Input(input_shape1)
    X1 = resnet_v2(X1_input, depth=11, num_classes=10)
    X1 = BatchNormalization()(X1)
    X2_input = Input(input_shape2)
    X2_input1 = Concatenate()([X1, X2_input])
    X2 = resnet_v2(X2_input1, depth=11, num_classes=10)
    X2 = BatchNormalization()(X2)
    X2 = Activation('softmax')(X2)
    model = Model(inputs=[X1_input, X2_input], outputs=X2)
    model._name = 'input_cascade_resnet_' + str(datetime.utcnow() + timedelta(hours=2)).split(".")[0].replace(":", "_").replace(" ", "_")
    return model

