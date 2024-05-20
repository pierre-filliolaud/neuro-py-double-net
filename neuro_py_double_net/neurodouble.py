from keras import layers

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

model = Sequential()

# units = nb neuro
# input
# Define model layers
model.add(layers.Dense(units=3, input_shape=[1]))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=1))

#Input/Output
input_data = [1,2,3,4,5]
output_data = [2,4,6,8,10]

# Convert list to numpy arrays
x = np.array(input_data)
y = np.array(output_data)

# Compile and train model
model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x=x,y=y,epochs=1000)
model.fit(x=x,y=y,epochs=2500)

# Save model
model.save('neural_network_model.keras')

# Load saved model
loaded_model = tf.keras.models.load_model('neural_network_model.keras')

# Re-create optimizerand compile loaded model
new_optimizer = tf.keras.optimizers.Adam()
loaded_model.compile(loss='mean_squared_error', optimizer=new_optimizer)

# Continue training of loaded trained model
additional_input_data = [6, 7, 8, 9, 10]
additional_output_data = [12, 14, 16, 18, 20]
x_additional = np.array(additional_input_data)
y_additional = np.array(additional_output_data)

loaded_model.fit(x=x_additional, y=y_additional, epochs=1000)


# Prediction loop
while True:
    z = int(input('Number: '))
    z_array = np.array([[z]])  # Convert to numpy array
    prediction = model.predict(z_array)
    print('Output: ' + str(prediction[0][0]))  # Display prediction