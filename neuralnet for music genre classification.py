import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "D:\COLLEGE\PROJECTS\Audio DL Basics\datasets\dataset.json"


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert lists into numpy arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets

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
        keras.layers.Dense(512, activation = 'relu'),

        # 2st hidden layer
        keras.layers.Dense(256, activation = 'relu'),

        # 3st hidden layer
        keras.layers.Dense(64, activation = 'relu'),

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
    model.fit(x_train, y_train,
              epochs = 50, 
              validation_split = 0.1,
              batch_size = 32)
    
    # training results-
    # loss: 0.1493 - accuracy: 0.9539 - val_loss: 2.9074 - val_accuracy: 0.5608
    