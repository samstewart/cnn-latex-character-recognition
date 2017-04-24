from keras.datasets import mnist
from kera.smodels import Sequential
from keras.layers import Dense, Activation

# load the MNIST data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()


# build the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# now train the model
# Q: what does this batch_size parameter mean?
model.fit(x_train, y_train, epochs=5, batch_size=32)

# now try to classify a few example images
# TODO: load one of the images from the file system
classes1 = model.predict(x_test, batch_size=128)
