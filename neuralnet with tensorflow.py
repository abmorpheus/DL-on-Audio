import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

def gen_dataset(num_samples, test_size):

    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
    return x_train, x_test, y_train, y_test



if __name__ == "__main__":
    x_train, x_test, y_train, y_test = gen_dataset(1000, 0.2)

    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim = 2, activation = 'sigmoid'), 
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
    model.compile(optimizer = optimizer, loss = 'MSE')

    # train model
    model.fit(x_train, y_train, epochs = 500)

    # evaluate model
    print()
    print('Model evaluation: ')
    model.evaluate(x_test, y_test, verbose = True)

    # making predictions
    input = np.array([[0.3, 0.4]])
    pred = model.predict(input)

    print(f'Sum of {input[0][0]} and {input[0][1]} predicted by model is {pred[0][0]}')



