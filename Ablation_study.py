import ipykernel 	# 進度條格式更改
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from readDataset import *
from attention import *
from loss import CE
from transformer import vision_transformer_block
import tensorflow as tf
from tensorflow.keras.layers import *
from layer import GroupNormalization
import resnet_def_copy


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(rotation_range=20,horizontal_flip=True,height_shift_range=shift,width_shift_range=shift)
    datagen.fit(x_train)
    return datagen


def define_model(input_shape=(100, 100, 1), classes=7):


	inputLayer, outLayer, scaleLayer = resnet_def_copy.ResNet50(input_shape=input_shape, include_top=False, create_encoder=True, weights='imagenet')

	x = Conv2D(64, (3, 3), padding='same')(outLayer)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)

	x = Conv2D(64, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	y1 = x = Activation(gelu)(x)
	x = Conv2D(64, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)
	x = concatenate([x, y1])

	y2 = x = Conv2D(128, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(128, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	y2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y2)
	x = concatenate([x, y2])

	y3 = x = Conv2D(256, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	# y3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y3)
	# x = concatenate([x, y3])

	y4 = x = Conv2D(512, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = concatenate([x, y4])

	x = Conv2D(512, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = GroupNormalization()(x)
	x = Activation(gelu)(x)

	x2 = Conv2D(128, (3, 3), padding='same')(outLayer)
	x2 = GroupNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x21 = x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
	x2 = Conv2D(128, (3, 3), padding='same')(x2)
	x2 = GroupNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x2 = Add()([x2, x21])
	x2 = Conv2D(256, (3, 3), padding='same')(x2)
	x2 = GroupNormalization()(x2)
	x2 = Activation(gelu)(x2)
	x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
	x2 = Conv2D(512, (3, 3), padding='same')(x2)

	x = Multiply()([x, x2])
	x = GlobalAveragePooling2D()(x)
	x = Dense(128, activation='gelu')(x)
	x = Dense(7, activation='softmax')(x)

	# Residual in Residual Dense Block
	RRDBlock = outLayer  # RRD_1
	RDBlock = RDBlocks(outLayer, name='RDBlock_1', count=3, g=32)
	RDBlock = RDBlocks(RDBlock, name='RDBlock_2', count=3, g=32)
	RDBlock = RDBlocks(RDBlock, name='RDBlock_3', count=3, g=32)
	RDBlock = RDBlocks(RDBlock, name='RDBlock_4', count=3, g=32)

	x3 = Add()([RRDBlock, RDBlock])  # RRD_1
	x3 = GlobalAveragePooling2D()(x3)
	x4 = vision_transformer_block(outLayer)
	x3 = Dense(7, activation='gelu')(x3)
	x4 = Dense(7, activation='gelu')(x4)
	x = Concatenate()([x, x3, x4])


	x = Dense(7, activation='softmax')(x)


	return models.Model(inputs=inputLayer, outputs=x)

def run_model():
	fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

	x_train, x_test, y_train, y_test, x_val, y_val = readFERplus()
	x_train, x_test, y_train, y_test = readRAFDB()
	datagen = data_augmentation(x_train)

	epochs = 400
	batch_size = 16

	# Training model from scratch

	black = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
	black.summary()
	black.compile(optimizer=Adam(learning_rate = 0.0001), loss=[CE, categorical_crossentropy],loss_weights=[0.9, 0.1], metrics=['accuracy'])
	# history = black.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,steps_per_epoch=len(x_train) // batch_size,validation_data=(x_val, y_val), verbose=2)
	history = black.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, steps_per_epoch=len(x_train) // batch_size, validation_data=(x_test, y_test), verbose=2)
	test_loss, test_acc = black.evaluate(x_test, y_test, batch_size=batch_size)

	plot_acc_loss(history)
	save_model_and_weights(black, test_acc)

def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
    plt.show()

def RDBlocks(x, name, count=6, g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=gelu,
                 name=name + '_conv1')(x)

    for i in range(2, count + 1):
        li.append(pas)
        out = Concatenate(axis=3)(li)  # conctenated output
        pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=gelu,
                     name=name + '_conv' + str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis=3)(li)
    feat = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=gelu,
                  name=name + '_Local_Conv')(out)
    feat = Add()([feat, x])
    return feat


def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    plt.show()


def save_model_and_weights(model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000+50000)
    model_json = model.to_json()
    with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize and save weights to JSON
    model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')

def gelu(x):
	cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
	return x * cdf

if __name__ == '__main__':
	run_model()
