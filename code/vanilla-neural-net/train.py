# trains a simple neural network on the latex images
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

# the size of an image
IMAGE_SIZE = (400, 400)

model.add(Dense())
# use the usual max{ } activation
model.add(Activation('relu'))