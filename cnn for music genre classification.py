import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "D:\COLLEGE\PROJECTS\Audio DL Basics\datasets\dataset.json"
NUM_CLASSES = 10


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert lists into numpy arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets


def prepare_datasets(test_size, val_size):

    # load dataset
    x, y = load_dataset(DATASET_PATH)

    # create train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

    # create train val split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size, random_state = 42)

    # create 3d array
    x_train = x_train[..., np.newaxis] # 4d array => (num_samples, 130, 13, 1)
    x_test = x_test[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(input_shape):

    # create model
    model = keras.Sequential()
    
    # 1st conv layer
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (2, 2), activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    return model



def plot_history(history):
    
    fig, axes = plt.subplots(2)

    # accuracy subplot
    axes[0].plot(history.history['accuracy'], label = 'train accuracy')
    axes[0].plot(history.history['val_accuracy'], label = 'test accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc = 'lower right')
    axes[0].set_title('Accuracy eval')

    # error subplot
    axes[1].plot(history.history['loss'], label = 'train error')
    axes[1].plot(history.history['val_loss'], label = 'test error')
    axes[1].set_ylabel('Error')
    axes[1].set_xlabel('Epochs')
    axes[1].legend(loc = 'upper right')
    axes[1].set_title('Eval eval')

    plt.show()


def predict(model, x, y):

    # x is 3d (130, 13, 1)
    # convert to 4d (1, 130, 13, 1)
    x = x[np.newaxis, ...]
    pred = model.predict(x)

    # pred is 2d array = [[p1, p2, p3, ... , p10]]
    # extract index with max value
    predicted_index = np.argmax(pred, axis = 1) 
    
    print(f'Expected index: {y} \nPredicted index: {predicted_index}')




if __name__ == '__main__':

    # prepare data
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.2, 0.1)

    print(f'x_train len: {len(x_train)}')
    print(f'x_val len: {len(x_val)}')
    print(f'x_test len: {len(x_test)}')

    print('\n\n')

    # build cnn architecture
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    model.summary()
    
    # train model
    history = model.fit(x_train, y_train,
              epochs = 30, 
              validation_data = (x_val, y_val),
              batch_size = 32)
    
    # model results
    # loss: 0.7917 - accuracy: 0.7237
    
    plot_history(history)

    # evaluate on test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose = True)
    print(f'Accuracy on test set: {test_accuracy}')

    
    # make prediction on a sample
    x = x_test[567]
    y = y_test[567]
    predict(model, x, y)