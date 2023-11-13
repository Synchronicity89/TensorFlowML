import tensorflow as tf
import numpy as np
import time

def eq_1(x):
    return 0.5 * x ** 2 + 5 * x + 11

def eq_2(x):
    return 0.4 * x ** 2 + 9 * x + 7

# Data Generation Function
def generate_data(size=100000, range_x=(-10, 10), focus_range=(3, 5), focus_size=50000, noise=0.1):
    x_focus = np.linspace(*focus_range, focus_size).reshape(-1, 1)
    x_sparse = np.linspace(*range_x, size - focus_size).reshape(-1, 1)
    x_sparse = x_sparse[np.abs(x_sparse - 4) >= 1]
    x = np.concatenate([x_focus.reshape(-1, 1), x_sparse.reshape(-1, 1)], axis=0)
    y = np.where(x < 4, eq_1(x), eq_2(x))
    y += noise * np.random.randn(*y.shape)
    return x, y

def train():
    # Generate Data
    x_train, y_train = generate_data()

    # Convert to TensorFlow dataset and optimize for performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.cache().shuffle(buffer_size=100000).batch(2048).prefetch(tf.data.AUTOTUNE)

    # Model Definition
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Model
    start_time = time.time()
    model.fit(train_dataset, epochs=200, verbose=1)
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds")

    # Save and Load the Model for Testing
    model.save('my_model.h5')
    new_model = tf.keras.models.load_model('my_model.h5')

    # Testing
    test_inputs = np.array([[-5], [0], [5]])
    model_outputs = new_model.predict(test_inputs)
    expected_outputs = np.where(test_inputs < 4, eq_1(test_inputs), eq_2(test_inputs))

    for inp, out, exp in zip(test_inputs, model_outputs, expected_outputs):
        print(f"Input: {inp[0]}, Model Output: {out[0]}, Expected Output: {exp[0]}")
if __name__ == '__main__':
    train()  # Uncomment this line to train the model

    new_model = tf.keras.models.load_model('my_model.h5')
                                           
    import matplotlib.pyplot as plt

    # Generate test data from -15 to 15
    x_test = np.linspace(-15, 15, 300).reshape(-1, 1)
    y_actual = np.where(x_test < 4, eq_1(x_test), eq_2(x_test))

    # Predict using the model
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
                                     
