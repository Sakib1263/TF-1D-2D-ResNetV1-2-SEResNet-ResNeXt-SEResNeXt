# ResNet 1D-Convolution Architecture in Keras - For both Classification and Regression Problems
"""Reference: [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385)"""


from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Add, Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D


def Conv_1D_Block(inputs, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    conv = Conv1D(model_width, kernel, strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    batch_norm = BatchNormalization()(conv)
    activate = Activation('relu')(batch_norm)

    return activate


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7)
    if conv.shape[1] <= 2:
        pool = MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool


def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3)
    conv = Conv_1D_Block(conv, num_filters, 3)
    return conv


def residual_block(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv_1D_Block(inputs, num_filters, 3)
    conv = Conv_1D_Block(conv, num_filters, 3)
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
    if conv.shape[1] <= 2:
        pool = MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
    return pool


def conv_block_bottleneck(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3)
    conv = Conv_1D_Block(conv, num_filters, 3)
    conv = Conv_1D_Block(conv, num_filters, 3)
    return conv


def residual_block_bottleneck(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 4, 1)
    #
    conv = Conv_1D_Block(inputs, num_filters, 1)
    conv = Conv_1D_Block(conv, num_filters, 3)
    conv = Conv_1D_Block(conv, num_filters * 4, 1)
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


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = Dense(class_number, activation='softmax')(inputs)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = Dense(feature_number, activation='linear')(inputs)
    return out


class ResNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout=False, dropout_rate=0.2):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def MLP(self, x):
        outputs = []
        if self.pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = Flatten(name='flatten')(x)
        if self.dropout:
            x = Dropout(self.dropout_rate, name='Dropout')(x)
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)
        return outputs

    def ResNet18(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))      # The input tensor
        stem_ = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)               # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model

    def ResNet34(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))      # The input tensor
        stem_ = stem(inputs, self.num_filters)               # The Stem Convolution Group
        x = learner34(stem_, self.num_filters)               # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model

    def ResNet50(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner50(stem_b, self.num_filters)             # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model

    def ResNet101(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner101(stem_b, self.num_filters)            # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model

    def ResNet152(self):
        outputs = []
        inputs = Input((self.length, self.num_channel))     # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner152(stem_b, self.num_filters)            # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = Model(inputs, outputs)
        return model
