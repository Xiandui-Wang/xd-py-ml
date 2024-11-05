import numpy as np
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
import matplotlib.pyplot as plt


print(tf.__version__)

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)

input_layer1 = Dense(units=10, activation='sigmoid', input_dim=1)
hidden_layer2 = Dense(units=10, activation='relu')
hidden_layer3 = Dense(units=10, activation='relu')
output_layer = Dense(units=1)

model = Sequential([input_layer1, hidden_layer2, hidden_layer3, output_layer])

model.compile(optimizer="sgd", loss="mse")

model.fit(x, y, epochs=10000, verbose=2)

sin_func_result = model.predict(x)

plt.scatter(x, y)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('数据点散点图')
plt.show()