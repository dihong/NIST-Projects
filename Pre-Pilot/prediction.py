'''
The NIST Pre-Pilot event prediction project.

The codes are provided by the datascience research group at University of Florida (http://dsr.cise.ufl.edu).

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.
'''
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel
import csv
import os
import sys
from collections import Counter
from functools import partial
import time
import operator
from sklearn.linear_model import LinearRegression
import numpy as np

csv.field_size_limit(sys.maxsize)

def event_prediction(thres,min_num_points,b_events,tc_list):
	# b_events: [(x,y,type,year)]
	events = ['accidentsAndIncidents','roadwork','precipitation','deviceStatus','obstruction','trafficConditions']
	ret = []	# [(accidentsAndIncidents,roadwork,precipitation,deviceStatus,obstruction,trafficConditions)]
	lr = LinearRegression()
	for xmin,xmax,ymin,ymax in tc_list:
		cnt = Counter([(e_type,year) for x,y,e_type,year in b_events.value if x>xmin and x<xmax and y>ymin and y<ymax])  # {(e_type,year):count}
		counts = []
		for e in events:
			year_count = {key[1]:val for key,val in cnt.items() if key[0] == e}	# {year:count}
			if len(year_count) == 0:
				counts.append("0.00")
				continue
			year_count_desc_c = sorted(year_count.items(), key=operator.itemgetter(1), reverse = True)  # [(year,count)], decending by count.
			current_max = year_count_desc_c[0][1]
			train_points = []	 # (year,count)
			for y,c in year_count_desc_c:
				if c >= thres*current_max:
					current_max = c
					train_points.append((y,c))
			if len(train_points) < min_num_points:
				# most recent year data for prediction
				year_count_desc_y = sorted(train_points, key=operator.itemgetter(0), reverse = True)  # [(year,count)], decending by year.
				counts.append("%.2f"%(year_count_desc_y[0][1]/12.0))	# use the most recent year for prediction, because we don't have sufficient samples for model training.
			else:
				# linear regression for prediction
				x = np.array([v[0] for v in train_points])
				y = np.array([v[1] for v in train_points])
				m = lr.fit(x[:, np.newaxis], y)
				counts.append("%.2f"%(m.predict(2015)[0]/12.0))
		ret.append(counts)
	return ret

if __name__=="__main__":

	# import configurations
	from config import cfg
	
	# setup Spark
	conf = (SparkConf()
		.setMaster(cfg.SYS.SPARK_MASTER)
		.set("spark.app.name","NIST Pre-Pilot Prediction Task"))
	sc = SparkContext(conf = conf)
	
	# verify files
	assert os.path.isfile('data/events_train.csv'), "Error: you have to place the data/events_train.csv file to the script folder."
	assert os.path.isfile('data/prediction_trials.tsv'), "Error: you have to place the data/prediction_trials.tsv file to the script folder."

	# read testing cases
	with open('data/prediction_trials.tsv','rb') as csvfile:
		tc = [(float(r[1]),float(r[0]),float(r[3]),float(r[2])) for r in csv.reader(csvfile, delimiter='\t')]	# [(xmin,xmax,ymin,ymax)]
	
	# load training data.
	with open('data/events_train.csv', 'rb') as csvfile:
		rows = [r for r in csv.reader(csvfile, delimiter=',') if r[10].find('.') > -1]
		event_rows = [(float(r[10]),float(r[9]),r[6],int(r[4].split('-')[0])) for r in rows]	# [(x,y,type,year)]
	print "Load %d rows from data/events_train.csv" % len(event_rows)
	
	# broadcast event data
	b_events = sc.broadcast(event_rows)
	
	# parallelize testing bounding boxes
	rdd_tc = sc.parallelize(tc,len(tc)).coalesce(cfg.SYS.CORES).glom()
	
	# predict #events
	predictions = rdd_tc.map(partial(event_prediction,cfg.PREDICTION.dec_thres,cfg.PREDICTION.min_points_reg,b_events)).reduce(lambda x,y: x+y)
	outfile = "prediction_out.txt"
	with open (outfile,"w+") as f:
		lines = []
		for t in predictions:
			lines.append("\t".join(t))
		f.write("\n".join(lines))
	print "Done. Results saved to %s" % outfile
	
