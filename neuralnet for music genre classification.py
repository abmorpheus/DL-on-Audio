import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "D:\COLLEGE\PROJECTS\Audio DL Basics\datasets\dataset.json"


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert lists into numpy arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets


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





if __name__ == '__main__':

    # load data
    inputs, targets = load_dataset(DATASET_PATH)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(inputs, 
                                                        targets, 
                                                        test_size = 0.3, 
                                                        random_state = 42)
    print(f'x_train len: {len(x_train)}')
    print(f'y_train len: {len(y_train)}')
    print('\n\n')
    # build network architecture
    model = keras.models.Sequential([
        # input layer
        keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])), # (interval, n-mfcc)

        # 1st hidden layer
        # , kernel_regularizer = keras.regularizers.l2(0.001)
        keras.layers.Dense(512, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.3),

        # 2st hidden layer
        keras.layers.Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.3),

        # 3st hidden layer
        keras.layers.Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation = 'softmax')
    ])

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, 
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    model.summary()
    
    # train model
    history = model.fit(x_train, y_train,
              epochs = 50, 
              validation_split = 0.1,
              batch_size = 32)
    
    plot_history(history)
    
    # training results (without drop and regularization) -
    # loss: 0.0527 - accuracy: 0.9849 - val_loss: 2.8389 - val_accuracy: 0.5880
    # leads to overfitting

    # handled overfitting
    # loss: 2.6400 - accuracy: 0.2424 - val_loss: 2.5912 - val_accuracy: 0.2432
    