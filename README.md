Project Report
Titanic Survival Prediction



Introduction

The aim was to design various Machine Learning models based on a given training data set (the famous Titanic dataset) and test them on the test dataset to predict the survival of passengers and compare the various models for accuracy. 



Summary

Using the Titanic Survival Kaggle dataset, I determined the best Machine Learning Model based on their Accuracy scores after K Fold Validation.
I performed Feature Engineering on the dataset and then built four ML Models- Logistic Regression, Random Forest, Support Vector Machine and K Nearest Neighbours. 
I used GridSearchCV for parameter tuning for each of the models and then trained the ML models on the Kaggle dataset using these parameters.
I plotted the Confusion Matrix and calculated Accuracy, Recall and Precision. 
I used K Fold Validation to test the Accuracy Score for each model and compared them.
The best model was SVM with an accuracy of 0.947.



Steps Followed

Data Collection

The data was collected from the Kaggle Dataset of Titanic Survival Records. There were three datasets: train, test and gender_submission. The train dataset contained the training data for Titanic Survival Prediction, test dataset contained the test data and the gender_submission contained the actual data of whether the passengers in test data survived or not.



Data Cleaning

The data was cleaned by forming new features from existing ones and then dropping the old ones, the NA values were replaced by either average of other values in the same column or the most frequently used value. Important or relevant information was extracted from irrelevant columns before dropping them. Categories were created in the Age and Fare columns.



Visualisation

For visualisation, the following graphs were plotted:
Age vs Survival Rate graphs for both the genders
Embarked vs Survived vs Pclass vs Sex
Pclass vs Survived Bar Graph
Survived vs Pclass vs Age Histogram



Machine Learning Models

The Machine Learning Models created were:
Logistic Regression
Support Vector Machine
Random Forest
K Nearest Neighbours
For each model, GridSearchCV was used for hyperparameter tuning and then the models were trained on the ‘Train’ dataset using the best parameters.



Evaluation

The ML Models were evaluated on the basis of their Accuracy Score after K Fold Validation using cross_val_score. 
The Precision, Recall and Accuracy of each individual model was also calculated. 
Confusion Matrix was plotted for every model.



Conclusion

The best Machine Learning Model for given Titanic Survival Dataset was Support Vector Machines with an accuracy of 0.947.



Further Study

More ML models are to be added to the project, including but not limited to: Naive Bayes, Stochastic Gradient Descent and Decision Tree.

