# SE-ResNet models for Keras.
# Reference for ResNets - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))
# Reference for SE-Nets - [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf))


import tensorflow as tf


def Conv_1D_Block(inputs, num_filters, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(num_filters, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, num_filters])(excitation)

    scale = inputs * excitation

    return scale


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3, 2)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)

    return conv


def residual_block(inputs, num_filters, ratio):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv_1D_Block(inputs, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = SE_Block(conv, num_filters, ratio)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group(inputs, n_filters, ratio, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for _ in range(n_blocks):
        out = residual_block(out, n_filters, ratio)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, n_filters * 2)

    return out


def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def residual_block_bottleneck(inputs, num_filters, ratio):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 4, 1, 1)
    #
    conv = Conv_1D_Block(inputs, num_filters, 1, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters * 4, 1, 1)
    conv = SE_Block(conv, num_filters * 4, ratio)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group_bottleneck(inputs, n_filters, ratio, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for _ in range(n_blocks):
        out = residual_block_bottleneck(out, n_filters, ratio)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, n_filters * 2)

    return out


def learner18(inputs, num_filters, ratio):
    # Construct the Learner
    x = residual_group(inputs, num_filters, ratio, 2)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, ratio, 1)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, ratio, 1)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, ratio, 1, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner34(inputs, num_filters, ratio):
    # Construct the Learner
    x = residual_group(inputs, num_filters, ratio, 3)          # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, ratio, 3)           # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, ratio, 5)           # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, ratio, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner50(inputs, num_filters, ratio):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, ratio, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, ratio, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, ratio, 5)   # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, ratio, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner101(inputs, num_filters, ratio):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, ratio, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, ratio, 3)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, ratio, 22)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, ratio, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner152(inputs, num_filters, ratio):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, ratio, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, ratio, 7)   # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, ratio, 35)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, ratio, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out


def SqueezeExcite(x, ratio=16):
    nb_chan = tf.keras.backend.int_shape(x)[-1]
    y = tf.keras.layers.GlobalAveragePooling1D()(x)
    y = tf.keras.layers.Dense(nb_chan // ratio, activation='relu')(y)
    y = tf.keras.layers.Dense(nb_chan, activation='sigmoid')(y)
    y = tf.keras.layers.Multiply()([x, y])

    return y


class SEResNet:
    def __init__(self, length, num_channel, num_filters, ratio=16, problem_type='Regression', output_nums=1,
                 pooling='avg', dropout_rate=False):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate

    def MLP(self, x):
        outputs = []
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name='flatten')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        # Problem Types
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs


    def SEResNet18(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters, self.ratio)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def SEResNet34(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner34(stem_, self.num_filters, self.ratio)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def SEResNet50(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner50(stem_b, self.num_filters, self.ratio)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def SEResNet101(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner101(stem_b, self.num_filters, self.ratio)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def SEResNet152(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner152(stem_b, self.num_filters, self.ratio)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model


if __name__ == '__main__':
    # Configurations
    length = 1024
    model_name = 'SEResNet152'  # Modified DenseNet
    model_width = 16  # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Channels in the Model
    problem_type = 'Regression' # Classification or Regression
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    reduction_ratio = 4
    # Build, Compile and Print Summary
    Model = SEResNet(length, num_channel, model_width, ratio=reduction_ratio, problem_type=problem_type,
                     output_nums=output_nums, pooling='avg', dropout_rate=False).SEResNet152()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
