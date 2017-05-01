# Pulls stroke data from original format and renders it to images with a given width and height.
# TODO: would be nice to know the maximum frame size for the sample strokes. Can we determine this from
# the online source code?
"""
A few test samples in the test_samples.json file
"""

from format_fname import format_filename
from PIL import Image, ImageDraw
from pymongo import MongoClient
from bson.objectid import ObjectId
from numpy import array, min, max, concatenate
from pprint import pprint
import os

# the symbols we want to learn
symbols = ['latex2e-OT1-_alpha',
'latex2e-OT1-_theta',
'latex2e-OT1-_tau',      
'latex2e-OT1-_beta',
'latex2e-OT1-_vartheta',
'latex2e-OT1-_pi',
'latex2e-OT1-_upsilon',
'latex2e-OT1-_gamma',
'latex2e-OT1-_gamma',
'latex2e-OT1-_varpi',
'latex2e-OT1-_phi',
'latex2e-OT1-_delta',
'latex2e-OT1-_kappa',
'latex2e-OT1-_rho',
'latex2e-OT1-_varphi',
'latex2e-OT1-_epsilon',
'latex2e-OT1-_lambda',
'latex2e-OT1-_varrho',
'latex2e-OT1-_chi',
'latex2e-OT1-_varepsilon',
'latex2e-OT1-_mu',
'latex2e-OT1-_sigma',
'latex2e-OT1-_psi',
'latex2e-OT1-_zeta',
'latex2e-OT1-_nu',
'latex2e-OT1-_varsigma',
'latex2e-OT1-_omega',
'latex2e-OT1-_eta',
'latex2e-OT1-_xi',
'latex2e-OT1-_Gamma',
'latex2e-OT1-_Lambda',
'latex2e-OT1-_Sigma',
'latex2e-OT1-_Psi',
'latex2e-OT1-_Delta',
'latex2e-OT1-_Xi',
'latex2e-OT1-_Upsilon',
'latex2e-OT1-_Omega',
'latex2e-OT1-_Theta',
'latex2e-OT1-_Pi',
'latex2e-OT1-_Phi']

def bounding_rectangle(points):
	"""
	:param points a nx2 array of points (n is number of points)
	:return 
	"""
	points = array(points)[:, 0:2]

	# assuming (0, 0) at origin (kinda tricky line)
	# simply find the min x, y and the max x, y
	lower_left_corner = min(points, 0)
	upper_right_corner = max(points, 0)

	return concatenate((lower_left_corner, upper_right_corner))

def stroke_to_image(image_drawer, stroke, scale, shift):

	"""
	image: the ImagerDraw context for drawing the lines
	stroke: an entry from the 'strokes' array from the MongoDB database
	"""
	
	# we ignore the time information column
	# affine transformation to center the image and change the coordinate orientation 
	stroke = array(stroke)[:, 0:2].astype('float64')
	stroke -= shift
	# flip the y coordinates
	#stroke[:, 1] *= -1
	stroke *= scale 

	# TODO: convert the drawing to some kind of monad thing. We need added context to make this clean.
	points = map(lambda p: (p[0], p[1]), stroke) # need format for the line drawer

	image_drawer.line(points, fill="black", width=1)


def convert_to_image(sample, mongo):
	# first find the bounding rectangle 
	# we ignore the time stamp nonsense
	# find the bounding rectangle across all strokes

	strokes = sample['strokes']

	# each row is a bounding rectangle (lower left | upper right)
	bounding_rects = array(map(lambda s: bounding_rectangle(s), strokes))
				
	# now we find the biggest bounding rectangle
	# mins of all the lower left corners
	# maxes of all the upper left corners
	final_lower_left = min(bounding_rects[:, 0:2], 0)
	final_upper_right = max(bounding_rects[:, 2:4], 0)

	size_of_region = final_upper_right - final_lower_left	
	print(sample['classified_latex_code'])	
	print(size_of_region)
	
	if max(size_of_region) == 0:
		return
	# scale to fit in 28x28 frame
	scale = 28.0 / max(size_of_region)
	
		# L denotes black and white (who the fuck would have guessed that?) I assume it stands for 'long' but my god, just stupid.
	im = Image.new('L', (28, 28), "white")
	image_drawer = ImageDraw.Draw(im)

	# actually draw the strokes
	map(lambda s: stroke_to_image(image_drawer, s, scale, final_lower_left), strokes)

	# export the image relative to root 
	sample['image'] = "data/image-samples/%s.png" % (sample['_id'],)

	im.save('../../' + sample['image'])

	# save back to the database
	mongo.detexify.samples.save(sample)

	return sample['image']

def extract_images():
	"""
	Renders all stroke patterns to images.
	"""
	MONGO_URL = os.environ.get('MONGO_URL')

	if not MONGO_URL:
		MONGO_URL = "mongodb://localhost:27017"

	mongo = MongoClient(host=MONGO_URL)

	samples = mongo.detexify.samples.find({'classified_latex_code' : {"$in" : symbols }})
	# convert a few samples to images
	for s in samples:
		convert_to_image(s, mongo)

	mongo.close()

if __name__ == "__main__":
	extract_images()	
