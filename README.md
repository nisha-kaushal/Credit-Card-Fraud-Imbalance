# Detecting Credit Card Fraud with Imbalanced Data: Oversampling vs. Undersampling

### Goal: 
Credit card fraud detection using labeled data is one of the main basic projects used while learning about data science, machine learning, and binary classifiers. In this notebook, I am less focused on building simple classification models to detect fraud, and instead, I am curious to test the difference between oversampling and undersampling techniques and their outcomes when applied to highly-imbalanced data, like the dataset used here. Utilizing the Imbalanced-Learn library, my goal is to find which of the two sampling techniques is more useful in highly imbalanced data, and how each affects the accuracy.
### Data Source: 
The data can be found on Kaggle, here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud <br> 
(note that the file itself exceeds GitHub's limit) <br> 

Boxplot for all transactions: <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/d6513100-c3d4-4101-843c-3391c37ca06a)


Boxplot for transactions up to $500
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/7384804e-05b5-4fa6-b31b-e291f50b6c5a)

Boxplot for non-fraud vs fraud <br> 
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/01619f2f-1c62-420f-9581-1f6912d98b26)

Correlation Matrix <br> 
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/1cc8550f-6c1e-406b-b1ec-9db198091f80)

Transactions over time <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/c1d32239-ecf4-4330-a824-768aa41b66b9)


Day 1 vs. Day 2 amount of fraud transactions <br> 
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/66e613eb-f48b-4b02-8cdb-c3526fbb9a2c)

Confusion Matrix, Random Forest with Oversampling <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/0260119c-9719-434a-a2e1-9636cddf7e35)

Confusion Matrix, Random Forest with Undersampling <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/d3e6e239-4aa8-4d97-85ad-3d29669c00f3)


Confusion Matrix, K-Nearest Neighbors with Oversampling <br>
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/21cf9ef9-c152-4c6a-8b4a-15da7fbb71c3)

Confusion Matrix, K-Nearest Neighbors with Undersampling <br> 
![image](https://github.com/nisha-kaushal/Credit-Card-Fraud-Imbalance/assets/100887571/2605536a-a163-41ee-9d64-fd9e1c6653bf)




