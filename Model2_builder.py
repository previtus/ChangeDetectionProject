# [ Siamese Segmentation models ]
#
# Altered code from:
# https://github.com/qubvel/segmentation_models
# more specifically combined from files:
# - https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/unet/builder.py
# - https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/unet/blocks.py
# under commit https://github.com/qubvel/segmentation_models/commit/9c68d81d66e4fb856770a87b450a43bb2ae6ddba

from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model

from segmentation_models.utils import freeze_model
from segmentation_models.utils import legacy_support
from segmentation_models.backbones import get_backbone, get_feature_layers

from segmentation_models.unet.blocks import Transpose2D_block
from segmentation_models.utils import get_layer_number, to_tuple

from keras.layers import Concatenate
from segmentation_models.unet.blocks import UpSampling2D, handle_block_names, ConvRelu

import keras
from keras.layers import Input
from keras.models import load_model

old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'skip_connections': 'encoder_features',
    'upsample_rates': None,  # removed
    'input_tensor': None,  # removed
}


@legacy_support(old_args_map)
def SiameseUnet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         classes=1,
         activation='sigmoid',
         encoder_weights='imagenet',
         encoder_freeze=False,
         encoder_features='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         **kwargs):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

        Args:
            backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
            classes: a number of classes for output (output shape - ``(h, w, classes)``).
            activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
            encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
                layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
            decoder_block_type: one of blocks with following layers structure:

                - `upsampling`:  ``Upsampling2D`` -> ``Conv2D`` -> ``Conv2D``
                - `transpose`:   ``Transpose2D`` -> ``Conv2D``

            decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
            decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.

        Returns:
            ``keras.models.Model``: **Unet**

        .. _Unet:
            https://arxiv.org/pdf/1505.04597

    """

    load_weights_from = None
    if encoder_weights is not "imagenet" and encoder_weights is not None:
        load_weights_from = encoder_weights
        encoder_weights = None


    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)

    if load_weights_from is not None:
        model_to_load_weights_from = load_model(load_weights_from)

        # now let's assume that this loaded model had its own "top" upsampling section trained on another task
        # let's transplant what we can, that is the backbone encoder

        output = model_to_load_weights_from.layers[len(backbone.layers)-1].output  # remove activation and last conv layer
        transplant = keras.models.Model(model_to_load_weights_from.input, output)
        #transplant.summary()

        transplant.save("transplant.h5") # hacky way
        backbone.load_weights("transplant.h5")

        # Check if the weights have been loaded
        """
        inspect_i = 0
        import numpy as np
        w1 = np.asarray(transplant.get_weights()[inspect_i])
        print(w1)
        w2 = np.asarray(backbone.get_weights()[inspect_i])
        print(w2)
        """
        print("Loaded weights into ",backbone_name,"from",load_weights_from)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=4)

    model = build_siamese_unet(backbone,
                       classes,
                       encoder_features,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=len(decoder_filters),
                       upsample_rates=(2, 2, 2, 2, 2),
                       use_batchnorm=decoder_use_batchnorm,
                       input_shape=input_shape)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'u-{}'.format(backbone_name)

    return model



def Siamese_Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip_a=None, skip_b=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip_a is not None and skip_b is not None:
            x = Concatenate()([x, skip_a, skip_b]) # siamese concatenation

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


def build_siamese_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True,
               input_shape=(None, None, 3)):

    verbose = False
    if verbose:
        print("Entered build_unet with arguments:")
        print("backbone",backbone)
        #print("---\n")
        #backbone.summary()
        #print("---\n")


        print("classes",classes)
        print("skip_connection_layers",skip_connection_layers)
        print("decoder_filters",decoder_filters)
        print("upsample_rates",upsample_rates)
        print("n_upsample_blocks",n_upsample_blocks)
        print("block_type",block_type)
        print("activation",activation)
        print("use_batchnorm",use_batchnorm)

    input = backbone.input
    x = backbone.output

    # Prepare for multiple heads in siamese nn:

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    if verbose:
        print("skip_connection_idx", skip_connection_idx)

    skip_connections = []
    for idx in skip_connection_idx:
        skip_connection = backbone.layers[idx].output
        skip_connections.append(skip_connection)

    if verbose:
        print("skip_connections layers", len(skip_connections), skip_connections)
    #4 layers
    # 'stage4_unit1_relu1/Relu:0' shape=(?, 16, 16, 256)
    # 'stage3_unit1_relu1/Relu:0' shape=(?, 32, 32, 128)
    # 'stage2_unit1_relu1/Relu:0' shape=(?, 64, 64, 64)
    # 'relu0/Relu:0'              shape=(?, 128, 128, 64)

    siamese_backbone_model_encode = Model(inputs=[input], outputs=[x]+skip_connections)

    if verbose:
        print("siamese_model_encode.input", siamese_backbone_model_encode.input)
        print("siamese_model_encode.output", siamese_backbone_model_encode.output) # x and the (now 4) skip connections

    # Then merging
    input_a = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    input_b = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    branch_a_outputs = siamese_backbone_model_encode([input_a])
    branch_b_outputs = siamese_backbone_model_encode([input_b])

    branch_a = branch_a_outputs[0]
    branch_b = branch_b_outputs[0]

    x = Concatenate(name="concatHighLvlFeat")([branch_a, branch_b]) # both inputs, in theory 8x8x512 + 8x8x512 -> 8x8x1024

    skip_connection_outputs_a = branch_a_outputs[1:]
    skip_connection_outputs_b = branch_b_outputs[1:]

    if block_type == 'transpose':
        up_block = Transpose2D_block
        assert False # NOT IMPLEMENTED
    else:
        up_block = Siamese_Upsample2D_block

    for i in range(n_upsample_blocks):
        skip_connection_a = None
        skip_connection_b = None
        if i < len(skip_connection_idx): # also till len(skip_connection_outputs_a)
            skip_connection_a = skip_connection_outputs_a[i]
            skip_connection_b = skip_connection_outputs_b[i]

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip_a=skip_connection_a, skip_b=skip_connection_b, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    #model = Model(input, x)
    full_model = Model(inputs=[input_a, input_b], outputs=x)

    return full_model



# There is support for all of these (with weights from ImageNet included) ... qubvel/segmentation_models is awesome!
# VGG           'vgg16' 'vgg19'
# ResNet	    'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
# SE-ResNet	    'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
# ResNeXt	    'resnext50' 'resnet101'
# SE-ResNeXt	'seresnext50' 'seresnet101'
# SENet154	    'senet154'
# DenseNet	    'densenet121' 'densenet169' 'densenet201'
# Inception	    'inceptionv3' 'inceptionresnetv2'
# MobileNet	    'mobilenet' 'mobilenetv2'
# Performance comparison for classification: https://github.com/qubvel/classification_models

"""
BACKBONE = 'resnet34'

custom_weights_file = "model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5" # None
custom_weights_file = "imagenet"

model = SiameseUnet(BACKBONE, encoder_weights=custom_weights_file, classes=3, activation='softmax', input_shape=(256, 256, 3))
print("Model loaded:")
print("model.input", model.input)
print("model.output", model.output)
"""
#model.summary()

# Ps: there is posibility to change the code of additional models in similar manner to get FPN, Linknet and PSPNet
# Ps2: some of these Siamese NN models end up with large amount of parameters ...
#      if we don't have much data, we should perhaps freeze some of the layers of the encoder... "encoder_freeze=False"

# Ps3: keras saves models into $ cd /home/<username>/.keras/models/