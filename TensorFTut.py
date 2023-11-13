import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

import numpy as np

def eq_1(x):
    return np.sin(10 * x) + np.sin(5 * x) + 0.1 * x ** 2

def eq_2(x):
    return np.sin(15 * x) + np.exp(0.2 * x) - 10

# Data Generation Function
def generate_data(size=1000000, range_x=(-10, 10), focus_range=(3.6, 9.5), focus_size=50000, noise=0.1):
    x_focus = np.linspace(*focus_range, focus_size).reshape(-1, 1)
    x_sparse = np.linspace(*range_x, size - focus_size).reshape(-1, 1)
    x_sparse = x_sparse[np.abs(x_sparse - 4) >= 1]
    x = np.concatenate([x_focus.reshape(-1, 1), x_sparse.reshape(-1, 1)], axis=0)
    y = np.where(x < 4, eq_1(x), eq_2(x))
    y += noise * np.random.randn(*y.shape)
    return x, y

def train():
    x_train, y_train = generate_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.cache().shuffle(buffer_size=1000000).batch(1024*4).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20/2, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(40/2, activation='relu'),
        tf.keras.layers.Dense(60/2, activation='relu'),
        tf.keras.layers.Dense(40/2, activation='relu'),
        tf.keras.layers.Dense(20/2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    start_time = time.time()
    model.fit(train_dataset, epochs=400, verbose=1)
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds")

    model.save('my_model.h5')

if __name__ == '__main__':
    train()  # Uncomment to train the model

    new_model = tf.keras.models.load_model('my_model.h5')
                                           
    # Generate test data from -15 to 15
    x_test = np.linspace(-15, 15, 300).reshape(-1, 1)
    y_actual = np.where(x_test < 4, eq_1(x_test), eq_2(x_test))
    y_predicted = new_model.predict(x_test)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_actual, color='blue', label='Actual')
    plt.scatter(x_test, y_predicted, color='red', alpha=0.7, label='Predicted')
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
