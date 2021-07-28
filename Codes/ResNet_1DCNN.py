# Import Necessary Libraries
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Add, Dense
from keras.layers import BatchNormalization, Activation


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv1D(num_filters, 7, strides=2, padding='same', kernel_initializer="he_normal")(inputs)
    pool = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv)
    return pool


def conv_block(inputs, n_filters):
    # Construct Block of Convolutions without Pooling
    # inputs        : input into the block
    # n_filters: number of kerels or filters
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(n_filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def residual_block(inputs, n_filters):
    # Construct a Residual Block of Convolutions
    # inputs   : input into the block
    # n_filters: number of kerels or filters
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
    # inputs   : input to the group
    # n_filters: number of kernels or filters
    # n_blocks : number of blocks in the group (varies accross ResNet Models)
    # conv     : flag to include the convolution block connector
    out = []
    for _ in range(n_blocks):
        out = residual_block(inputs, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    if conv:
        out = conv_block(out, n_filters * 2)
    return out


def learner18(inputs, num_filters):
    # Construct the Learner for ResNet18
    x = residual_group(inputs, num_filters, 2)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 1)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 1)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 1, False)  # Fourth Residual Block Group of 512 filters
    return out


def learner34(inputs, num_filters):
    # Construct the Learner for ResNet34
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
    pool = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv)
    return pool


def conv_block_bottleneck(inputs, n_filters):
    # Construct Block of Convolutions without Pooling - BottleNeck Structure for ResNet50, ResNet101 and ResNet152
    # Read the Paper for More Information
    # inputs   : input into the block
    # n_filters: number of kernels or filters
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
    # Construct a Residual Block of Convolutions - BottleNeck Structure
    # inputs   : input into the block
    # n_filters: number of kernels or filters
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
    # inputs   : input to the group
    # n_filters: number of filters or kernels
    # n_blocks : number of blocks in the group, varies among ResNet Models
    # conv     : flag to include the convolution block connector
    out = []
    for _ in range(n_blocks):
        out = residual_block_bottleneck(inputs, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    if conv:
        out = conv_block_bottleneck(out, n_filters * 2)
    return out


def learner50(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3) # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5)  # Third Residual Block Group of 256 filters
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


def classifier(inputs, n_classes):
    # Construct the Classifier
    # inputs         : input vector
    # n_classes : number of output classes
    # Pool at the end of all the convolutional residual blocks
    x = GlobalAveragePooling1D()(inputs)
    # Final Dense Outputting Layer for the outputs
    out = Dense(n_classes, activation='softmax')(x)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor
    # inputs         : input vector
    # n_classes : number of output features
    # Pool at the end of all the convolutional residual blocks
    x = GlobalAveragePooling1D()(inputs)
    # Final Dense Outputting Layer for the outputs
    out = Dense(feature_number, activation='linear')(x)
    return out

class ResNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression', output_nums=1024):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters # num_filters = 64 [Default]
        self.problem_type = problem_type
        self.output_nums = output_nums
        
    def ResNet18(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))  # The input tensor
        x = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner18(x, self.num_filters)               # The learner
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet34(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))  # The input tensor
        x = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner34(x, self.num_filters)               # The learner
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet50(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))   # The input tensor
        pool = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner50(pool, self.num_filters)             # The learner
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet101(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))   # The input tensor
        pool = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner101(pool, self.num_filters)            # The learner
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model


    def ResNet152(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))   # The input tensor
        pool = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner152(pool, self.num_filters)            # The learner
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model
