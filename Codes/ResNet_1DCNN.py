# ResNet 1D-Convolution Architecture in Keras - For both Classification and Regression Problems
"""Reference: [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385)"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Add, Dense
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv1D(num_filters, 7, strides=2, padding='same', kernel_initializer="he_normal")(inputs)
    if (conv.shape[1] <= 2):
        pool = MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool


def conv_block(inputs, n_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def residual_block(inputs, n_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv1D(n_filters, 3, strides=1, padding="same", kernel_initializer="he_normal")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=1, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Add()([conv, shortcut])
    out = Activation('relu')(conv)
    return out


def residual_group(inputs, n_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = []
    for _ in range(n_blocks):
        out = residual_block(inputs, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, n_filters * 2)
    return out


def learner18(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 2)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 1)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 1)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 1, False)  # Fourth Residual Block Group of 512 filters
    return out


def learner34(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 3)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 3)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 5)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out


def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv1D(num_filters, 7, strides=2, padding='same', kernel_initializer="he_normal")(inputs)
    if (conv.shape[1] <= 2):
        pool = MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool


def conv_block_bottleneck(inputs, n_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def residual_block_bottleneck(inputs, n_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv1D(n_filters * 4, 1, strides=1, padding="same", kernel_initializer="he_normal")(inputs)
    shortcut = BatchNormalization()(shortcut)
    #
    conv = Conv1D(n_filters, 1, strides=1, padding="same", kernel_initializer="he_normal")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=1, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters * 4, 1, strides=1, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Add()([conv, shortcut])
    out = Activation('relu')(conv)
    return out


def residual_group_bottleneck(inputs, n_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = []
    for _ in range(n_blocks):
        out = residual_block_bottleneck(inputs, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block_bottleneck(out, n_filters * 2)
    return out


def learner50(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5)   # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out


def learner101(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 22)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out


def learner152(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 7)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 35)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters
    return out


def classifier(inputs, n_classes, pooling):
    # Construct the Classifier Group
    # x         : input vector
    # n_classes : number of output classes
    # Pool at the end of all the convolutional residual blocks
    x = []
    if pooling == 'avg':
        x = GlobalAveragePooling1D()(inputs)
    elif pooling == 'max':
        x = GlobalMaxPooling1D()(inputs)
    # Final Dense Outputting Layer for the outputs
    out = Dense(n_classes, activation='softmax')(x)
    return out


def regressor(inputs, feature_number, pooling):
    # Construct the Regressor Group
    # x         : input vector
    # n_classes : number of output classes
    # Pool at the end of all the convolutional residual blocks
    x = []
    if pooling == 'avg':
        x = GlobalAveragePooling1D()(inputs)
    elif pooling == 'max':
        x = GlobalMaxPooling1D()(inputs)
    # Final Dense Outputting Layer for the outputs
    out = Dense(feature_number, activation='linear')(x)
    return out


class ResNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression', output_nums=1, pooling='avg'):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
    def ResNet18(self):
        # num_filters = 64 [Default]
        outputs = []
        inputs = Input((self.length, self.num_channel))      # The input tensor
        stem_ = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)               # The learner
        pooling = self.pooling
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number, pooling)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number, pooling)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet34(self):
        # num_filters = 64 [Default]
        outputs = []
        inputs = Input((self.length, self.num_channel))      # The input tensor
        stem_ = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner34(stem_, self.num_filters)               # The learner
        pooling = self.pooling
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number, pooling)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number, pooling)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet50(self):
        # num_filters = 64 [Default]
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner50(stem_b, self.num_filters)             # The learner
        pooling = self.pooling
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number, pooling)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number, pooling)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet101(self):
        # num_filters = 64 [Default]
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner101(stem_b, self.num_filters)            # The learner
        pooling = self.pooling
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number, pooling)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number, pooling)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet152(self):
        # num_filters = 64 [Default]
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner152(stem_b, self.num_filters)            # The learner
        pooling = self.pooling
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number, pooling)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number, pooling)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model
