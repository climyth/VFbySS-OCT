from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import InceptionV4

def GetModel(base_model_name='InceptionResnet'):
    # model build ================================
    base_model = None
    if base_model_name == "InceptionResnet":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "InceptionV4":
        base_model = InceptionV4.create_model(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    if base_model == None:
        return None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 54 classes
    predictions = Dense(52, activation='relu')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def GetModel10(base_model_name='ResNet'):
    # model build ================================
    base_model = None
    if base_model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "DenseNet":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "NASNet":
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "InceptionResnet":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(200, 480, 3))
    elif base_model_name == "InceptionV4":
        base_model = InceptionV4.create_model(weights='imagenet', include_top=False, input_shape=(200, 480, 3))

    if base_model == None:
        return None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 54 classes
    predictions = Dense(68, activation='relu')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model