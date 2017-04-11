import urllib2
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

fname = 'data/all_employee_records_umn.csv'

# Sample entry
# {
# 'College Code': 'TCOA', 
# 'Empl Class': 'Temp Casual', 
# 'First Name': 'Nancy', 
# 'Department': 'Arboretum Educational Programs', 
# 'DeptID': '11049', 
# 'Start Date at U': '04/15/2002', 
# 'Annual Rt': '217.360',
#  'Job Entry Date': '04/15/2002', 
#  'ZDeptID Descr': 'CFANS Landscape Arboretum', 
#  'Job Code': '0001', 
#  'Last': 'Bjerke', 
#  'ZDeptID':  'Z0022', 
#  'Job Title': 'Non-Exempt Temporary or Casual', 
#  'FTE Percentage': '0.01', 
#  'Comp Rate': '16.720000', 
#  'Std Hrs': '0.25', 
#  'College Descr': 'FOOD, AGRI/NAT RSRC SCI, COLL'
#  }

counter = 0
columns_to_ignore = ['Empl Record', 
'ID', 
'Group', 
'Type', 
'Comp Freq', 
'Eff Date', 
'Pay Status', 
'Status Flag', 
'Action', 
'Reason', 'Tenure St', 'UM Tenure Stat', 'Tenure Home', 'UM Instl Email']

total_imported = 0
with open(fname, 'r') as csvfile:
	csv_reader = DictReader(csvfile, delimiter=',')

	for row in csv_reader:
		converted_row = {}

		# force some type conversions
		row['DeptID'] = int(row['DeptID'])
		row['Annual Rt'] = float(row['Annual Rt'])
		row['FTE Percentage'] = float(row['FTE Percentage'])
		row['Comp Rate'] = float(row['Comp Rate'])
		row['Std Hrs'] = float(row['Std Hrs'])
		row['Start Date at U'] = datetime.datetime.strptime(row['Start Date at U'], '%m/%d/%Y')
		row['Job Entry Date'] = datetime.datetime.strptime(row['Job Entry Date'], '%m/%d/%Y')

		# fix the names of the other columns
		for key, val in row.iteritems():
			if key not in columns_to_ignore:
				# convert spaces to underscores and uppercase to lower case
				new_key = key.replace(" ", "_").lower()
				converted_row[new_key] = row[key]

		# now put this into mongodb
		mongo.umn.employees.insert(converted_row)
		total_imported = total_imported + 1

print "Total imported records %d" % (total_imported, )
# fetch all the route information
# routes = json.loads(urllib2.urlopen(route_url).read())
# mongo.buses.routes.insert_many(routes)



# # now grab a location update from the route information
# for route in routes:
# 	locations = json.loads(urllib2.urlopen(locations_url % (int(route[u'Route']), )).read())
	
# 	if len(locations):
# 		mongo.buses.locations.insert_many(locations)

mongo.close()