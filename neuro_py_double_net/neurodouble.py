from keras import layers

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Prediction loop
while True:
    z = int(input('Number: '))
    z_array = np.array([[z]])  # Convert to numpy array
    prediction = model.predict(z_array)
    print('Output: ' + str(prediction[0][0]))  # Display prediction