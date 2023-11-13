import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class DiscontinuityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DiscontinuityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.discontinuity_point = self.add_weight(
            name='discontinuity_point',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(value=4.0),  # Initial value set near the expected discontinuity
            trainable=True)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        discontinuity_point = tf.cast(self.discontinuity_point, tf.float32)
        below = tf.sin(5.0 * inputs) * tf.cast(inputs < discontinuity_point, tf.float32)
        above = tf.sin(8.0 * inputs) * tf.cast(inputs >= discontinuity_point, tf.float32)
        return below + above

    def get_config(self):
        config = super().get_config()
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
            DiscontinuityLayer(input_shape=(1,)),  # Use the custom DiscontinuityLayer
            tf.keras.layers.Dense(20, activation='relu'),
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

    new_model = tf.keras.models.load_model('my_model.h5', custom_objects={'DiscontinuityLayer': DiscontinuityLayer})
                                           
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


    # After training
    discontinuity_layer = new_model.layers[0]  # Assuming the DiscontinuityLayer is the first layer
    discontinuity_point = discontinuity_layer.get_weights()[0]  # Retrieve the discontinuity point weight

    # Report the discontinuity point weight
    print("Discontinuity Point Weight:")
    print(discontinuity_point)

    # Visualize the Discontinuity Layer Activation
    test_inputs = np.linspace(-15, 15, 300).reshape(-1, 1)
    discontinuity_activation = discontinuity_layer(test_inputs)

    # Plot the activation pattern
    plt.figure(figsize=(10, 6))
    plt.plot(test_inputs, discontinuity_activation)
    plt.title('Discontinuity Layer Activation Pattern')
    plt.xlabel('Input Value')
    plt.ylabel('Layer Activation')
    plt.axvline(x=discontinuity_point, color='grey', linestyle='--', label=f'Discontinuity Point ~ {discontinuity_point[0]:.2f}')
    plt.legend()
    plt.show()
