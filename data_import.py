import json
from pprint import pprint
from pymongo import MongoClient
import os
from csv import DictReader
import datetime

MONGO_URL = os.environ.get('MONGO_URL')

if not MONGO_URL:
	MONGO_URL = "mongodb://localhost:27017"

mongo = MongoClient(host=MONGO_URL)

total_imported = 0
with open('data/detexify.csv', 'r') as csvfile:
	csv_reader = DictReader(csvfile, delimiter=',')

	for row in csv_reader:
		converted_sample = {}

		
		# now put this into mongodb
		mongo.detexify.samples.insert(
			{
				'classified_latex_code' : row['key'],
				'strokes' : json.loads(row['strokes'])
			})
		
		total_imported = total_imported + 1

print "Total imported records %d" % (total_imported, )

mongo.close()