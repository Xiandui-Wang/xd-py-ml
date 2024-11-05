import keras
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
import matplotlib.pyplot as plt

print(tf.__version__)

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x)

input_layer1 = Dense(units=1, activation='tanh', input_dim=1)
hidden_layer2 = Dense(units=10, activation='tanh')
hidden_layer3 = Dense(units=10, activation='tanh')
output_layer = Dense(units=1)

model = Sequential([input_layer1, hidden_layer2, hidden_layer3, output_layer])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

model.fit(x, y, epochs=3000, verbose=2)

sin_func_result = model.predict(x)

plt.scatter(x, sin_func_result)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('sin graph')
plt.show()
