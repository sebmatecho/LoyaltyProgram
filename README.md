# Loyalty Program Technical Assesment.

In this repository, a solution for a technical assesment used by a Colombian bank hiring Data Scientist is proposed. 

## The presented scenario

In such bank, a loyalty program is held once a year. When selected, customers will receive a formal invitation containing the details of the package while some benefits are offered. The bank would like to be very effective when selecting the clients to make the offer (this is, the highest the percentage of acceptance of the program, the better) and, thus, is relying on information gathered on a past campaings. Provided dataset contains customers financial information (anonymous information). We are tasked to build a model able to provide a list of potential clients to send the offer. 

## Data Exploration
In general, initial dataset was initially delivered considerably clean. There were no missing values and only one outlier was found.
##

## The final product

Once the best model was selected, a dashboard was deployed to assist decision-nmaking on potential new customers for the program. The final product is fully deployed in Streamlit and AWS (S3 for model and transformation). Final App available [here](https://share.streamlit.io/sebmatecho/loyaltyprogram/app_loyalty.py)
