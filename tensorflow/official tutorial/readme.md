# Keras API

tensorflow2.0 使用keras 接口来加载数据，分离数据，构建模型，训练模型

```python
# import module
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# 1. load and explore data, split dataset
f_m = keras.datasets.fashion_mnist
(x_, y_), (x, y) = f_m.load_data()

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# 2. build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # transfor the data into 1-D array
    keras.layers.Dense(128, activation='relu'),  # fully connect layer, has 128 neurons
    keras.layers.Dense(10)   # 10 classes means 10 neurons
])

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()  # check layers

## compile: loss, opitimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer='adam',  # optimizer = tf.keras.optimizers.RMSprop(0.001)
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy']) # metrics=['mae', 'mse'])

# 3. fit model
history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=22,  
                   batch_size=512,   # verbose=1, detial; =0 no message output
                   validation_data=(x_val, y_val), verbose=1)

model.fit(x_, y_, epochs=12)

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[PrintDot()])         

# 4. evaluate model and predict
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
results = model.evaluate(test_data,  test_labels, verbose=2)

## visualize the loss
import matplotlib.pyplot as plt
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
# “bo” for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b for blue line
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## predict
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x)

# 5. save and load the whole model 
model.save('my_model.h5')  # svae in format HDF5
new_model = keras.models.load_model('my_model.h5')  # load the model

# then u can do the same things
```

