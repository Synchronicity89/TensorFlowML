import tensorflow as tf
import numpy as np
import time

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    print("Name:", gpu.name, " Type:", gpu.device_type)


# Data Generation Function
def generate_data(size=100000, range_x=(-10, 10), focus_range=(3, 5), focus_size=50000, noise=0.1):
    x_focus = np.linspace(*focus_range, focus_size)
    x_sparse = np.linspace(*range_x, size - focus_size)
    x_sparse = x_sparse[np.abs(x_sparse - 4) >= 1]
    x = np.concatenate([x_focus.reshape(-1, 1), x_sparse.reshape(-1, 1)], axis=0)
    y = np.where(x < 4, 0.5 * x ** 2 + 6 * x + 8, 0.4 * x ** 2 + 7 * x + 9)
    y += noise * np.random.randn(*y.shape)
    return x, y

# Generate Data
x_train, y_train = generate_data()

# Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=256, verbose=1)
training_time = time.time() - start_time
print(f"Training time: {training_time} seconds")

# Save the Model
model.save('my_model.h5')

# Load and Test the Model
new_model = tf.keras.models.load_model('my_model.h5')
test_inputs = np.array([[-5], [0], [5]])
model_outputs = new_model.predict(test_inputs)
expected_outputs = np.where(test_inputs < 4, 0.5 * test_inputs ** 2 + 6 * test_inputs + 8, 0.4 * test_inputs ** 2 + 7 * test_inputs + 9)

for inp, out, exp in zip(test_inputs, model_outputs, expected_outputs):
    print(f"Input: {inp[0]}, Model Output: {out[0]}, Expected Output: {exp[0]}")
