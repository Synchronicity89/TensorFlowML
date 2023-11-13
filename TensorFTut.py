import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class SinusoidalLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(SinusoidalLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.frequency = self.add_weight(
            name='frequency',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True)
    
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        frequency = tf.cast(self.frequency, tf.float32)
        return tf.sin(tf.matmul(inputs, frequency * 4))
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

def eq_1(x):
    return np.sin(5 * x) + np.sin(5 * x) + 0.1 * x ** 2

def eq_2(x):
    return np.sin(8 * x) + np.exp(0.2 * x) - 10

# Data Generation Function
def generate_data(size=100000, range_x=(-10, 10), focus_range=(3.6, 4.4), focus_size=50000, noise=0.1):
    if size < focus_size:
        raise ValueError("size must be greater than or equal to focus_size")
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
    train_dataset = train_dataset.cache().shuffle(buffer_size=10000).batch(1024*4).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        SinusoidalLayer(20, input_shape=(1,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
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

    new_model = tf.keras.models.load_model('my_model.h5', custom_objects={'SinusoidalLayer': SinusoidalLayer})
                                           
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


# ... (existing code)

# After training
sinusoidal_layer = new_model.layers[0]  # Assuming the SinusoidalLayer is the first layer
frequency_weights = sinusoidal_layer.get_weights()[0]  # Retrieve the frequency weights

# Report the frequency weights
print("Frequency Weights:")
print(frequency_weights)

# Visualize the frequency weights as a histogram
plt.figure(figsize=(10, 6))
plt.hist(frequency_weights.flatten(), bins=50, alpha=0.75)
plt.title('Histogram of Frequency Weights')
plt.xlabel('Frequency Value')
plt.ylabel('Count')
plt.show()

# Test the Sinusoidal Layer Activation
test_inputs = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)  # Inputs over two periods
sinusoidal_activation = sinusoidal_layer(test_inputs)  # Pass inputs through the SinusoidalLayer

# Plot the activation pattern
plt.figure(figsize=(10, 6))
plt.plot(test_inputs, sinusoidal_activation)
plt.title('Sinusoidal Layer Activation Pattern')
plt.xlabel('Input Value')
plt.ylabel('Layer Activation')
plt.show()
