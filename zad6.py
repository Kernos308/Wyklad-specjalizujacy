import tensorflow as tf
from tensorflow.keras.datasets import mnist
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from IPython.display import Image


print(tf.reduce_sum(tf.random.normal([1000, 1000])))

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape

y_train.shape

X_test.shape

y_test.shape
sns.set(font_scale=2)

index = np.random.choice(np.arange(len(X_train)),24, replace= False)
figure, axes = plt.subplots(nrows = 4, ncols = 6, figsize=(16,9))
for axes, image, target in zip(axes.ravel(), X_train[index], y_train[index]):
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.set_title(target)
    plt.tight_layout()



X_train = X_train.reshape((60000,28,28,1))
X_train.shape
X_test = X_test.reshape((10000,28,28,1))
X_test.shape

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_train.shape

y_train[0]

y_test = to_categorical(y_test)
y_test.shape

cnn = Sequential()


cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", input_shape = (28, 28, 1)))
cnn.add(MaxPooling2D(pool_size= (2, 2)))


cnn.add(Conv2D(filters = 128, kernel_size = (3,3), activation = "relu"))
cnn.add(MaxPooling2D(pool_size= (2, 2)))


cnn.add(Flatten())

cnn.add(Dense(units = 128, activation = "relu"))

cnn.add(Dense(units = 10, activation = "softmax"))

cnn.summary()

plot_model(cnn, to_file='convnet.png', show_shapes= True, show_layer_names = True)
Image(filename='convnet.png')

cnn.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(X_train, y_train, epochs = 5, batch_size = 64, validation_split = 0.1)
loss, accuracy = cnn.evaluate(X_test, y_test)
przypuszczenia = cnn.predict(X_test)

for indeks, przypuszczenie in enumerate(przypuszczenia[0]):
    print(f'{indeks}: {przypuszczenie:.10%}')


obrazy = X_test.reshape((10000, 28, 28))
chybione_prognozy = []
for i, (p, e) in enumerate(zip(przypuszczenia, y_test)):
    prognozowany, spodziewany = np.argmax(p), np.argmax(e)

    if prognozowany != spodziewany:
        chybione_prognozy.append((i, obrazy[i], prognozowany, spodziewany))

print(len(chybione_prognozy))

figure, axes = plt.subplots( nrows = 4, ncols = 6, figsize = (16,12))

for axes, element in zip(axes.ravel(), chybione_prognozy):
    indeks, obraz, prognozowany, spodziewany = element
    axes.imshow(obraz, cmap = plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(f'indeks: {indeks}\np: {prognozowany}; s: {spodziewany}')

plt.tight_layout()
