# Detecting Credit Card Fraud with Imbalanced Data: Oversampling vs. Undersampling

### Goal: 
Credit card fraud detection using labeled data is one of the main basic projects used while learning about data science, machine learning, and binary classifiers. In this notebook, I am less focused on building simple classification models to detect fraud, and instead, I am curious to test the difference between oversampling and undersampling techniques and their outcomes when applied to highly-imbalanced data, like the dataset used here. Utilizing the Imbalanced-Learn library, my goal is to find which of the two sampling techniques is more useful in highly imbalanced data, and how each affects the accuracy.
### Data Source: 
The data can be found on Kaggle, here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud <br> 
(note that the file itself exceeds GitHub's limit) <br> 

### Skills Used: 
* Python Programming
* Exploratory Data Analysis
* Supervised Learning Classification
* Visualization (Plotly) 

## Data Analysis
The first part of this project comprises of exploratory data analysis. 

To begin, I found the descriptive statistics of all transactions, which can be found in the boxplot below: <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/d6513100-c3d4-4101-843c-3391c37ca06a) <br>
The figure was created using Plotly, which allows for hover annotations. Hovering over the original figure will show the minumum, lower fence, first quarter, median,third quarter, and upper fence values. It seems that there are a significant amount of outliers, making it so that we cannot get an accurate visualization without removing a large portion of the outliers. To visualize the boxplot portion better, I then created a cutoff point for the y-axis range ($500):

![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/7384804e-05b5-4fa6-b31b-e291f50b6c5a) <br>
While the majority of the transactions are within the \\$0-185 range, there seems to be a wide range of transaction amounts above that amount. Are the larger-amount outliers within the fraud (Class = 1) transactions, or the normal (Class = 0) transactions?

![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/01619f2f-1c62-420f-9581-1f6912d98b26) <br>
It looks like the overall outliers come from the transactions that are deemed "normal" (non-fraud). Interestingly, however,the majority of the fraud transactions are between \\$0-262, a wider range than the normal transactions (\\$0-184). 

Now that I know a little bit about the amounts, I want to understand the other features a bit more. Because we do not know what these features entail, one thing I can look at is the correlation between them all, along with the two classes. 


![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/1cc8550f-6c1e-406b-b1ec-9db198091f80) <br>
In general, there is neutral correlation between all the "V_" features. When looking at the other 3 features (Class, Amount, and Time), we can see that there is a little more variation in correlations. For example, there is a highly positive correlation between the transaction amount and features V7 and V20, meaning that, generally, when the value of V7 (or V20) moves, the transaction amount moves in the same direction. There is a negative correlation between feature V2 and Amount, meaning that, a V2 increases, Amount tends to decrease, and vice versa. Aside from these few correlations, the majority of the correlations are about neutral.

Before going into the predictions, I wanted to see the trends for each transaction amount vs. time. Note that "Time" is the seconds between each of the transaction and the first transaction (example: a "2.0" Time value would mean the transaction occurred 2 seconds after the very first transation in the dataset). I will also add indicators for when full days pass, since the time goes up to about 173,000 seconds (2 days = 172,800 seconds)


![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/c1d32239-ecf4-4330-a824-768aa41b66b9) <br>
In the non-fraud data (class 0), there seems to be a dip in overall transactions between the 2.5 hours post-first transaction and the 7 hours post-first transaction on the first day, which then repeats on the second day. There is no discernable pattern like this in the fraud class (class 1), however there does seem to be less fraud transactions within the second day than the first day. To check this, I will compare the amount of fraud transactions within the first day vs. the amount of fraud transactions within the second day


![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/66e613eb-f48b-4b02-8cdb-c3526fbb9a2c) <br>
There were about 70 less fraud transactions on day 2 than day 1. 

## Oversampling vs. Undersampling
### Oversampling
**Oversampling** is adding synthetic data to the minority class, in order to make the amount of data evenly distributed (balanced) among each class. <br>
For Oversampling, I used the **Synthetic Minority Oversampling Technique** (SMOTE). Essentially, when applied, SMOTE looks into the k-nearest neighbors of the minority class, and chooses synthetic data based on those neighbors. SMOTE can be applied using the [imblearn library](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).<br> 
***Steps to implement SMOTE:*** <br> 
1. import imblearn.oversampling 
2. create a SMOTE object, using sampling_strategy = 'minority" 
3. fit the object to the data to get oversampled X values and oversampled Y values
4. concatenate the oversampled X's and Y's into one dataframe

After implementing, the both classes had ***284315 transactions (568630 transactions in total)***

### Undersampling 
**Undersampling** reduces the amount of data from the majority class, to match the amount of data of the minority class. 
There are several undersampling techniques that can be used. For this project, I used **Near Miss Undersampling**. There are 3 versions of NearMiss, and for the purpose of this notebook, I used Version 3, as described below: <br> 
***Version 3 of the Near Miss Undersampling keeps an example from the majority class for each closest record of the minority class. It should result in an increased accuracy, as it takes into consideration the  majority class values near the decision boundary***<br>

The steps to implement NearMiss is similar to implementing SMOTE, except NearMiss is imported from imblearn.undersampling. After implementing, the both classes had ***492 transactions (984 transactions in total)***.

For this project, I implemented 4 different classification algorithms to classify transactions as fraud, or non-fraud: Random Forest, Decision Trees, Logistic Regression, and K-Nearest Neighbors. All algorithms were implemented with the oversampled data and the undersampled data, using identical hyperparameters. 

As a result, **oversampling** resulted in 95%+ accuracy scores, recall scores, and precision scores for all models. Undersampling, on the otherhand, showed a decrease of the three metrics. 

Below is an example of differences between the oversampling and undersampling models for the Random Forest Classifiers and the K-Nearest Neighbors Classifiers: 

![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/0260119c-9719-434a-a2e1-9636cddf7e35)


![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/d3e6e239-4aa8-4d97-85ad-3d29669c00f3)



![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/21cf9ef9-c152-4c6a-8b4a-15da7fbb71c3)


![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/2605536a-a163-41ee-9d64-fd9e1c6653bf)

#### Findings
Note that confusion matrices were made for the other classifiers as well, and showed a similar amount of difference as Random Forest saw between sampling types. 

Comparing the Random Forest Classifiers, which were the strongest models overall, 0.0035% of the "normal" (non-fraud) transactions were incorrectly classified as "fraud" with the oversampling technique, and all the fraud transactions were correctly classified as fraud. In the undersampling model, 2.6% of the normal transactions were incorrectly classified as "fraud", while 8.4% of the fraud transactions were classified as "normal." 

Comparing the K-Nearest Neighbors Classifiers, the overall weakest models, 6.3% of the oversampling normal data were incorrectly classified as "fraud", and 2.9% of the fraud data were incorrectly classified as "normal". In the undersampling model, **39.2%** of the normal data were incorectly labeled as "fraud", and **44.7%** of the fraud data was incorrectly labeled as "normal". 

## Oversampling vs. Undersampling- Which to Choose? 




