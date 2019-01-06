# Walmart-Sales-Forecast
Forecasting the Sales Trend based on the historical data
This is a Kaggle problem and you can find the problem statement and dataset in the link: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

* I did the exploratory analysis using Tableau and you can check out the reports in the documentation 
* Also analysed the correlation between the features 
* Run the python script to prepocess the data and run the models and get the accuracy of each model
  * The workflow in the code is:
    1) I'm merging the train data and the features data based on store and department in each store
    2) Preprocess the missing data in features data set 
    3) Categorize the weeks based on holidays
    4) Generating features and lable data to run the models and train them with the train x and train y
    5) Running the test data on the model and checking the accuracy with test y and the predictions from the model on test x
* Compared the accuracy of the three models(Linear Regression model, Random Forest and Extra Tress Regression)
* Extra Tress Regression model is the most accurate model for thos model
