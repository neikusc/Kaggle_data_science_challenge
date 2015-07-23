import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.cross_validation import KFold
from sklearn import linear_model as LM
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RF



num_cores = 8; k_folds = 10; seed = 2015
BASE_DIR = "/Users/kien/Desktop/kagglequestion/"
INPUT_TRAIN = "TrainingDataset.csv"; INPUT_TEST = "TestDataset.csv"
OUTPUT_TEST = "TestDataset_out.csv"



class Regressor(object):
  
  def __init__(self):
    self.resp_labels = ['Outcome_M' + x for x in map(str, range(1,13))]
    self.weka_labels = ["Date_1","Quan_2","Quan_3","Quan_4","Quan_5","Quan_6","Quan_7","Quan_8",
                       "Quan_9","Quan_10","Quan_11","Quan_12","Quan_12","Quan_14","Cat_2","Cat_4",
                       "Cat_6","Cat_16","Quan_18","Cat_110","Cat_147","Cat_148","Cat_149","Cat_218",
                       "Cat_221","Cat_228","Cat_291","Cat_292","Cat_318","Cat_374","Cat_384","Cat_409",
                       "Cat_456","Quant_24","Cat_466","Cat_476","Cat_482","Cat_494","Cat_495","Cat_498"]
                       
    self.train = pd.DataFrame()
    self.test = pd.DataFrame()
    self.resp = pd.DataFrame()
    
  def data_prepare(self):
    raw_train = pd.DataFrame.from_csv(BASE_DIR + INPUT_TRAIN, index_col=False)
    raw_test = pd.DataFrame.from_csv(BASE_DIR + INPUT_TEST, index_col=False)
    # remove douplicate columns
    df0 = raw_train[ self.weka_labels ]
    df0 = df0.T.drop_duplicates().T
    # fill NA values with median values.
    self.train = df0.apply(lambda x: x.fillna(x.median()), axis=0)
    # extracting labels
    self.resp = raw_train[self.resp_labels].apply(lambda x: x.fillna(x.median()), axis=0)
    self.test = raw_test[self.train.columns]
    self.test = self.test.apply(lambda x: x.fillna(x.median()), axis=0)
  
  def single_outcome_prediction(self, response, regr):
    # 10-fold cross validation for single outcome
    cv = KFold(len(response), k_folds, shuffle=True, random_state=seed)
    y_pred = response.copy()
    
    results = []
    for tr_idx, vl_idx in cv:
      x_train, x_vali = self.train.ix[tr_idx], self.train.ix[vl_idx]
      y_train = response[tr_idx]
      regr.fit( x_train, np.log(y_train) )
      y_pred[vl_idx] = np.exp( regr.predict(x_vali) )
      y_test = np.exp( regr.predict(self.test) )
      
      results.append( y_test )
    single_sse = np.mean( (y_pred-response)**2 )
    # return sum of square errror for 10-fold cross validation and 
    # single outcome of test set
    return single_sse, np.array(results).mean(axis=0)
  
  def total_sse(self, clf, **kwargs):
    regr = clf(**kwargs)
    sse = []; outcome_test = pd.DataFrame()
    for lb in self.resp_labels:
      single_sse, outcome = self.single_outcome_prediction(self.resp[lb], regr)
      outcome_test[lb] = outcome
      sse.append( single_sse )
    
    raw_test = pd.DataFrame.from_csv(BASE_DIR + INPUT_TEST, index_col=False)
    output = outcome_test.join(raw_test)

    #sse = Parallel(n_jobs=num_cores)( delayed(self.single_outcome_prediction)(
    #               self.resp[lb], learning_rate=0.1) for lb in self.resp_labels )
    return sse, output
  
  def save_output(self, output):
    output.to_csv(BASE_DIR+OUTPUT_TEST)




if __name__ == "__main__":
  reg = Regressor()
  reg.data_prepare()
  sse, output = reg.total_sse(GBR, learning_rate=0.1)
  print 'Root mean squared error by GBR: ' np.mean(sse)**0.5
  # save output into *.csv file.
  reg.save_output(output)
  
  sse, _ = reg.total_sse(RF, n_estimators=50, min_samples_split=2)
  print 'Root mean squared error by RF: ' np.mean(sse)**0.5


