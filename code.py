import pandas as pd
from pandas import *
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

#fuction read train,test and features data
def getFTTData():
    # for accuracy un comment below line and comment train.csv
    # training_data = pd.read_csv('train_accuracy.csv')
    training_data = pd.read_csv('train.csv')
    training_data = training_data[training_data.Date >= '2011-11-11']

    # for accuracy un comment below line and comment test.csv
    testing_data = pd.read_csv('test.csv')
    testing_data = testing_data[testing_data.Date >= '2011-11-11']

    features_data = pd.read_csv('features.csv')
    features_data = features_data[features_data.Date >= '2011-11-04']
    features_data = features_data[['Store','Date','Temp','Fuel_Price','MD1','MD2','MD3','MD4','MD5','IsHoliday']] 
    return (features_data,testing_data,training_data)

# this function is to fethch train x and y and test x and y	
def get_train_test_data(training,testing,features,md):
    training = np.array(training)
    testing = np.array(testing)
    features = np.array(features)
    train_x,train_y,test_x,dates=[],[],[],[]
	# flag is 0 for train and 1 for test data
    train_x,train_y, test_x,dates = get_TT(train_x,train_y,test_x,dates,0,training,testing,features,md)#train_data
    train_x,train_y, test_x,dates = get_TT(train_x,train_y,test_x,dates,1,training,testing,features,md)#test_data
    return (train_x,train_y,test_x,dates)

# function to fill the missing features data and categorize the dates and identify the holidays
def get_TT(train_x,train_y,test_x,dates,flag,train,test,feature,markdown):    
    if (flag == 0):
        data = train
        data_x = train_x
        data_y = train_y
    else:
        data = test
        data_x = test_x 
    j = 0
    for i in range(len(data)):
        data_x.append([])
        if flag ==0:
            store,dept,date,sales,isholiday = data[i]
        else :   
            store,dept,date,dummy,isholiday = data[i]
        f = search_in_features(store,date,feature,markdown)
        if flag == 0 :
            data_y.append(sales)
        data_x[j] = list(f)
		#categorizing the date based on the holidays into weeks
        temp = date.split('-')
        yy,mm,dd = int(temp[0]),int(temp[1]),int(temp[2])
        previous_week = get_which_holiday(str( datetime.date(yy,mm,dd) - datetime.timedelta(days=7)))
        upcoming_week = get_which_holiday(str( datetime.date(yy,mm,dd) + datetime.timedelta(days=7)))
        current_week = get_which_holiday(date)
        data_x[j]  = data_x[j]+ previous_week+current_week+upcoming_week
        if flag == 1:
            dates.append(date)
        j += 1
    if flag == 0:
        train_x = data_x
        train_y = data_y
    else:
        test_x = data_x
        
        
    return (train_x,train_y,test_x,dates)

# function to process the fill the features data	
def search_in_features(store,date,features,md):
    for i in range(len(features)):
        if features[i][1] == date and features[i][0] == store :
			# runnig the loop for markdowns columns (5th to 9th column in features file) , if the data is missing fill it with the mean value that was calculated
            for j in range(4,9):
                if isnull(features[i][j]):
                    features[i][j] = md[j-4]
            return features[i][2:-1]

def ExtraTreesRegressors(train_x, train_y, test_x):
    return ExtraTreesRegressor(n_estimators=50, max_features='log2').fit(train_x, train_y).predict(test_x)

def RandomForestRegressors(train_x, train_y, test_x):
    return RandomForestRegressor(n_estimators=10).fit(train_x, train_y).predict(test_x)

def linear_model(train_x,train_y,test_x):
    return LinearRegression().fit(train_x,train_y).predict(test_x)

# this function is for replacing the missing data with the mean value	
def repl_nan_with_mean(trains):
    markd,res = [],[]
    markd.extend(((list(trains.MD1)),(list(trains.MD2)),(list(trains.MD3)),(list(trains.MD4)),(list(trains.MD5))))
    for eachmkd in markd:
        res.append(np.array([x for x in eachmkd if notnull(x)]).mean())
    return res

#function to categorite date based on holidays
def get_which_holiday(date):
   
    if date in ['2012-02-10','2013-02-08']:#super_bowl_holiday
        return [0,0,0,1]
    elif date in ['2012-09-07','2013-09-06']:#labour_holiday
        return [0,0,1,0]
    elif date in ['2012-11-23','2011-11-25','2013-11-29']:#thanks_giving
        return [0,1,0,0]
    elif date in ['2012-12-28','2011-12-30','2013-12-27']:#christmas
        return [1,0,0,0]
    else:
        return [0,0,0,0] #no_holiday


def file_writing(y,store,dept,dates):
    with open('linear_model.csv','a') as f:
        for i in range(len(y)):
            f.write('%s,%s,%s,%s\n'%(store,dept,dates[i],y[i]))


if __name__=="__main__":

#This part of the code is to merge the train data and the features data

#uncomment the below line if you are running the code for Random Trees Regression model
#     with open('RandomTreesRegressors.csv','wb') as f:
#uncomment the below line if you are running the code for Linear Regression model
#         with open('LinearRegression.csv','wb') as f:
#comment the below line if you are not running the code for ExtraTreesRegressors model
    with open('ExtraTreesRegressors.csv','wb') as f:
        f.write('Store,Dept,Dates,Weekly_Sales\n')
    feature,test,train = getFTTData()
	no_of_stores = train.Store.nunique()
	#loop for each store
    for i in range(1,no_of_stores):
        index = train.Store == i
        traindata = train[index]
        depts = set(traindata.Dept.values)
        depts = list(depts)

        index =test.Store == i
        testdata = test[index]
        dept_test = set(testdata.Dept.values)
        dept_test = list(dept_test)

        index = feature.Store == i
        featuredata = feature[index]
        
        # for each department in a store
        for dept in dept_test:
            if dept not in depts:
                index = testdata.Dept == dept
                tests = testdata[index]
                length = len(tests)
                dates = list(tests.Date)
                y=[0 for j in range(length)]
                file_writing(y,i,dept,dates)
               
        for dept in depts:
            markdown = repl_nan_with_mean(featuredata)
            train_x,train_y,test_x,dates = get_train_test_data(traindata[traindata.Dept == dept],testdata[testdata.Dept == dept],featuredata,markdown)
            if len(test_x) > 0:



		# test_y = linear_model (train_x,train_y,test_x)
		#test_y = RandomForestRegressors (train_x,train_y,test_x)

                    test_y = RandomForestRegressors(train_x,train_y,test_x)
                    file_writing(test_y,i,dept,dates)
