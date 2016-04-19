from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C


#
# System
#
__C.SYS = edict()
__C.SYS.SPARK_MASTER = "spark://node5:7077"  # master node of the Apache Spark cluster.
__C.SYS.CORES = 59

#
# Data Prediction
#
__C.PREDICTION = edict()
__C.PREDICTION.dec_thres = 0.3				# Used to reject outlier training samples.
__C.PREDICTION.min_points_reg = 3			# Used to ensure number of training samples is at least min_points_reg.









