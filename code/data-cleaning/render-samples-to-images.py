# Pulls stroke data from original format and renders it to images with a given width and height.
# TODO: would be nice to know the maximum frame size for the sample strokes. Can we determine this from
# the online source code?
"""
A few test samples in the test_samples.json file
"""

from PIL import Image, ImageDraw
from pymongo import MongoClient
from bson.objectid import ObjectId
from numpy import array
import os

MONGO_URL = os.environ.get('MONGO_URL')

# width of the rendered stroke (pixels)
STROKE_SAMPLE_WIDTH = 6

if not MONGO_URL:
	MONGO_URL = "mongodb://localhost:27017"

mongo = MongoClient(host=MONGO_URL)


def draw_circle(image_drawer, center, radius):
	"""
	Draws a black circle in the given context. Assumes (0, 0) is in upper left corner
	center: a numpy array with two elements
	radius: the size of the ellipse
	"""

	# two corner points for the bounding box
	top_left = (center[0] - radius, center[1] - radius)
	lower_right = (center[0] + radius, center[1] + radius)

	# TODO: write better image drawing library for python. absurd architecture. bounding box my ass.
	image_drawer.ellipse([top_left, lower_right], fill="black")

def strokes_to_image(image_drawer, strokes):

	# draw all the strokes to the image
	map(lambda s: stroke_to_image(image_drawer, s), strokes)

def stroke_to_image(image_drawer, stroke):

	"""

	image: the ImagerDraw context for drawing the lines
	stroke: an entry from the 'strokes' array from the MongoDB database
	"""
	points = stroke_to_points(stroke)

	# TODO: convert the drawing to some kind of monad thing. We need added context to make this clean.
	points = map(lambda p: (p[0], p[1]), points)

	image_drawer.line(points, fill="black", width=STROKE_SAMPLE_WIDTH)


def stroke_to_points(stroke):
	"""
	Convert a stroke from the database to a 2D numpy array.
	:return a 2D numpy array of representing the stroke from the database
	"""

	# project (delete time coordinate) to spatial points
	return array(map(lambda p: p[0:2], stroke))

def convert_to_image(sample):
	size = (400, 400)

	im = Image.new('L', size, "white")
	image_drawer = ImageDraw.Draw(im)

	# actually draw the strokes
	strokes_to_image(image_drawer, sample['strokes'])

	fname = "data/image-samples/%s.png" % (sample['_id'],)



	im.save('../../' + fname)

	# store the path to the image
	sample['image'] = fname

	# save back to the database
	mongo.detexify.samples.save(sample)

	del image_drawer
# the L flag stands for grayscale image


# we have an example with multiple strokes

# sample for \mathcal{M}
testing_sample = mongo.detexify.samples.find({'_id': ObjectId('58eafec846418d29c25588f9')}).next()

# sample for \mathcal{P}

testing_sample = mongo.detexify.samples.find({'_id': ObjectId("58eafec846418d29c25588f5")}).next()

samples = mongo.detexify.samples.find({})

# convert a few samples to images
for s in samples:
	convert_to_image(s)

mongo.close()