# load images and convert to numpy arrays

from format_fname import format_filename
from PIL import Image, ImageDraw
from pymongo import MongoClient
from bson.objectid import ObjectId
from numpy import array, min, max, concatenate, zeros, floor, asarray
from pprint import pprint
import os
from images import symbols
import numpy



def export_to_numpy():
	MONGO_URL = os.environ.get('MONGO_URL')

	if not MONGO_URL:
		MONGO_URL = "mongodb://localhost:27017"

	mongo = MongoClient(host=MONGO_URL)

	# total number of records for all combined symbols:
	# 28784
	# partition into two sets: training and test roughly 75% - 25%
	total_samples = mongo.detexify.samples.find({'classified_latex_code' : {"$in" : symbols }}).count()
	# total_sample = 200 # debugging
	total_training = int(total_samples * .75)

	print("Total samples: %d" %(total_samples, ))
	samples = mongo.detexify.samples.find({'classified_latex_code' : {"$in" : symbols }})

	x_train = zeros((total_training, 28, 28))
	y_train = zeros(total_training)
	x_test = zeros((total_samples - total_training, 28, 28))
	y_test = zeros((total_samples - total_training))

	train_samples = 0
	test_samples = 0
	# convert a few samples to images
	for s in samples:
		im = Image.open('../../' + s['image'])
		im.load()

		a = asarray(im, dtype='int32') / 255.0

		if train_samples < total_training:
			x_train[train_samples, :, :] = a
			y_train[train_samples] = symbols.index(s['classified_latex_code'])
			train_samples += 1
		else:
			x_test[test_samples, :, :] = a
			y_test[test_samples] = symbols.index(s['classified_latex_code'])
			test_samples += 1

	numpy.savez('../../data/greek_character_samples', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

if __name__ == "__main__":
	export_to_numpy()