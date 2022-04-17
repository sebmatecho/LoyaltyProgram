# Loyalty Program Technical Assesment.

In this repository, a solution for a technical assesment used by a Colombian bank hiring Data Scientist is proposed. 

## The presented scenario

In such bank, a loyalty program is held once a year. When selected, customers will receive a formal invitation containing the details of the package while some benefits are offered. The bank would like to be very effective when selecting the clients to make the offer (this is, the highest the percentage of acceptance of the program, the better) and, thus, is relying on information gathered on a past campaings. Provided dataset contains customers financial information (anonymous information). We are tasked to build a model able to provide a list of potential clients to send the offer. 

## Data Exploration
In general, initial dataset was initially delivered considerably clean. There were no missing values and only one outlier was found. Train data frame contains 2628 rows while 1550 rows were contained. Pandas profiling was used to explore the data and further study the relationship between variables

Some remarks: 
- Only four cities were considered and each represents ~25% of data. 
- Once outlier is removed, mean of salary is 1493587 with standart deviation 669294 (Colombian pesos)
- Number of dependants are evenly distribuited between 0, 1 and 2.
- Five-numbers summary (min, Q1, median, Q2, max) for Antiquity are: 11, 43, 82, 117, 120. So, the base contains a significant portion of antique clients. 
- Surprisingly, age, number of dependentants and wealth are not highly correlated with any other variable.
- In training dataset, approximately half of customers accepted the program.
- No normal distribution was observed in any variable. 
## Data preparation. 
Once columns were understood, transformations were put in place. MinMaxScaler was selected for all numerical values, as no normal variables were observed (standarized) and outliers were removed (RobustScaler). 
## Machine Learning models 
Once variables were transformed, various machine Learning models were proposed. 


| Model Name | Accuracy  |
|---|---|
| XGBoost  |  0.848 +/- 0.030 |
|  Random Forest	 |  0.713 +/- 0.025 |
| KNN	  | 0.678 +/- 0.024  |   
| Naive Bayes	  | 0.514 +/- 0.035  | 
| Baseline  |  0.508 +/- 0.002 | 
|  Logistic Regression	 | 0.508 +/- 0.002  | 


	
	



	



## The final product

Once the best model was selected, a dashboard was deployed to assist decision-nmaking on potential new customers for the program. The final product is fully deployed in Streamlit and AWS (S3 for model and transformation). It looks like as the following:

<img src="test.gif" width="1000" height="550" />


Final App available [here](https://share.streamlit.io/sebmatecho/loyaltyprogram/app_loyalty.py)
