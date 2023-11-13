import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Constants
SIZE = 200000
RANGE_X = (-10, 10)
FOCUS_RANGE = (3.6, 4.4)
FOCUS_SIZE = 100000
NOISE_LEVEL = 0.1
FREQUENCY_COUNT = 1024
BATCH_SIZE = 1024 * 1
EPOCHS = 10
FOURIER_SCALE = 10
DISCONTINUITY_INIT = 4.0  # Initial guess for the discontinuity point

# Custom Modulation Layer
class ModulationLayer(tf.keras.layers.Layer):
    def __init__(self, enabled=True, **kwargs):
        super(ModulationLayer, self).__init__(**kwargs)
        self.enabled = enabled
    
    def build(self, input_shape):
        self.modulation_param = self.add_weight(
            name='modulation_param',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True)
    
    def call(self, inputs):
        if self.enabled:
            return inputs * self.modulation_param
        else:
            return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'enabled': self.enabled})
        return config

# Equations
def eq_1(x):
    return 0.5 * np.tanh(x) + np.cos(x)

def eq_2(x):
    return np.log(np.abs(x) + 1) + 0.5 * x

# Fourier Feature Mapping
def fourier_feature_mapping(x, B):
    x_fourier = np.concatenate([np.cos(x * B.T), np.sin(x * B.T)], axis=-1)
    return x_fourier

# Data Generation Function
def generate_data(size, range_x, focus_range, focus_size, noise_level):
    if size < focus_size:
        raise ValueError("size must be greater than or equal to focus_size")
    x_focus = np.linspace(*focus_range, focus_size)
    x_sparse = np.linspace(*range_x, size - focus_size)
    x_sparse = x_sparse[np.abs(x_sparse - DISCONTINUITY_INIT) >= 1]
    
    # Concatenate and reshape
    x = np.concatenate([x_focus, x_sparse], axis=0).reshape(-1, 1)
    y = np.where(x < DISCONTINUITY_INIT, eq_1(x), eq_2(x))
    y += noise_level * np.random.randn(*y.shape)
    return x, y

# Generating random Fourier features
B = np.random.normal(size=(FREQUENCY_COUNT,)) * FOURIER_SCALE
x_train, y_train = generate_data(SIZE, RANGE_X, FOCUS_RANGE, FOCUS_SIZE, NOISE_LEVEL)
x_train_fourier = fourier_feature_mapping(x_train, B)

# Training Function
def train(use_modulation_layer):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_fourier, y_train))
    train_dataset = train_dataset.cache().shuffle(buffer_size=SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential()
    if use_modulation_layer:
        model.add(ModulationLayer(input_shape=(FREQUENCY_COUNT * 2,)))  # Adjust input shape for the ModulationLayer
    
    model.add(tf.keras.layers.Dense(20, activation='relu', input_shape=(FREQUENCY_COUNT * 2,)))
    model.add(tf.keras.layers.Dense(40, activation='relu'))
    model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.Dense(40, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    start_time = time.time()
    history = model.fit(train_dataset, epochs=EPOCHS, verbose=1)
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds")

    model.save('my_model.h5')
    return model, history

# Main execution
if __name__ == '__main__':
    # Train with the ModulationLayer enabled
    model_with_modulation, history_with_modulation = train(use_modulation_layer=True)
    
    # Train without the ModulationLayer
    model_without_modulation, history_without_modulation = train(use_modulation_layer=False)
    
    # Generate test data and apply Fourier mapping
    x_test = np.linspace(-15, 15, 300).reshape(-1, 1)
    x_test_fourier = fourier_feature_mapping(x_test, B)
    y_actual = np.where(x_test < DISCONTINUITY_INIT, eq_1(x_test), eq_2(x_test))
    
    # Predict with both models
    y_predicted_with_modulation = model_with_modulation.predict(x_test_fourier)
    y_predicted_without_modulation = model_without_modulation.predict(x_test_fourier)

    # Plotting for model with modulation
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_actual, color='blue', label='Actual')
    plt.scatter(x_test, y_predicted_with_modulation, color='red', alpha=0.7, label='Predicted with Modulation')
    plt.title('Comparison with Modulation Layer')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Plotting for model without modulation
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_actual, color='blue', label='Actual')
    plt.scatter(x_test, y_predicted_without_modulation, color='green', alpha=0.7, label='Predicted without Modulation')
    plt.title('Comparison without Modulation Layer')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    # Plot training loss for both models
    plt.figure(figsize=(10, 6))
    plt.plot(history_with_modulation.history['loss'], label='With Modulation')
    plt.plot(history_without_modulation.history['loss'], label='Without Modulation')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
