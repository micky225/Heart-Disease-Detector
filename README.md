# AI and Data Documentation
## Prediction Model Documentation 

## Overview
This Python script contains code for developing a machine learning model to detect heart disease based on a given dataset. The model leverages various Python libraries for data preprocessing, feature engineering, model training, and evaluation.

## Purpose
The purpose of this model is to predict the presence or absence of heart disease in individuals based on a set of features. Early detection of heart disease can significantly improve patient outcomes by enabling timely interventions and treatments.

## Packages Used
The following packages are imported and used within the script:a

1. warnings: Python module for controlling warning messages.
2. numpy (np): Fundamental package for scientific computing with Python.
3. pandas (pd): Data manipulation and analysis library.
4. matplotlib.pyplot (plt): Plotting library for creating static, animated, and interactive visualizations.
5. seaborn (sns): Data visualization library based on matplotlib, providing a high-level interface for drawing attractive statistical graphics.
6. matplotlib.colors.ListedColormap: Module for creating a colormap from a list of colors.
7. sklearn.model_selection.train_test_split: Function for splitting arrays or matrices into random train and test subsets.
8. scipy.stats.boxcox: Function for computing the Box-Cox power transformation.
9. sklearn.pipeline.make_pipeline: Function for constructing a pipeline from the given estimators.
10. sklearn.preprocessing.StandardScaler: Class for standardization, scaling features by removing the mean and scaling to unit variance.
11. sklearn.neighbors.KNeighborsClassifier: K-Nearest Neighbors classifier implementation.
12. sklearn.svm.SVC: Support Vector Classification implementation.
13. sklearn.model_selection.GridSearchCV: Grid search with cross-validation for optimizing hyperparameters of a classifier.
14. sklearn.model_selection.StratifiedKFold: Stratified K-Folds cross-validator.
15. sklearn.metrics.classification_report: Function for building a text report showing the main classification metrics.
16. sklearn.metrics.accuracy_score: Function for computing the accuracy classification score.
17. sklearn.tree.DecisionTreeClassifier: Decision Tree classifier implementation.
18. sklearn.ensemble.RandomForestClassifier: Random Forest classifier implementation.
19. sklearn.linear_model.LogisticRegression: Logistic Regression classifier implementation.
20. sklearn.model_selection.cross_val_score: Function for evaluating a score by cross-validation.
21. joblib: Library for pipelining Python jobs.
22. pickle: Python object serialization library.

## Dataset 
### Dataset Description:

#### Variable Description

- **age**: Age of the patient in years

- **sex**: Gender of the patient (0 = male, 1 = female)

- **cp**: Chest pain type:
    - 0: Typical angina
    - 1: Atypical angina
    - 2: Non-anginal pain
    - 3: Asymptomatic

- **trestbps**: Resting blood pressure in mm Hg

- **chol**: Serum cholesterol in mg/dl

- **fbs**: Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false)

- **restecg**: Resting electrocardiographic results:
    - 0: Normal
    - 1: Having ST-T wave abnormality
    - 2: Showing probable or definite left ventricular hypertrophy

- **thalach**: Maximum heart rate achieved during a stress test

- **exang**: Exercise-induced angina (1 = yes, 0 = no)

- **oldpeak**: ST depression induced by exercise relative to rest

- **slope**: Slope of the peak exercise ST segment:
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping

- **ca**: Number of major vessels (0-4) colored by fluoroscopy

- **thal**: Thallium stress test result:
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect
    - 3: Not described

- **target**: Heart disease status (0 = no disease, 1 = presence of disease)

### Data Information
#### Inferences:

##### Number of Entries:
- The dataset consists of 1025 entries, ranging from index 0 to 1024.

##### Columns:
- There are 14 columns in the dataset corresponding to various attributes of the patients and results of tests.

##### Data Types:
- Most of the columns (13 out of 14) are of the int64 data type.
- Only the oldpeak column is of the float64 data type.

##### Missing Values:
- There don't appear to be any missing values in the dataset as each column has 303 non-null entries.

9 columns (sex, cp, fbs, restecg, exang, slope, ca, thal, and target) are numerical in terms of data type, but categorical in terms of their semantics. These features will be converted to string (object) data type for proper analysis

![Capture1](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/283f4b81-ecd1-4591-aa4d-9481bd83e855)

### Summary Statistics for Numerical Variables
#### Numerical Features:

- **age**: 
  - The average age of the patients is approximately 54.4 years, with the youngest being 29 and the oldest 77 years.

- **trestbps**: 
  - The average resting blood pressure is about 131.61 mm Hg, ranging from 94 to 200 mm Hg.

- **chol**: 
  - The average cholesterol level is approximately 246.00 mg/dl, with a minimum of 126 and a maximum of 564 mg/dl.

- **thalach**: 
  - The average maximum heart rate achieved is around 149.11, with a range from 71 to 202.

- **oldpeak**: 
  - The average ST depression induced by exercise relative to rest is about 1.07, with values ranging from 0 to 6.2.


### Summary Statistics for Categorical Variables
#### Categorical Features (object data type):

- **sex**: 
  - There are two unique values, with females (denoted as 1) being the most frequent category, occurring 713 times out of 1025 entries.

- **cp**: 
  - Four unique types of chest pain are present. The most common type is "0", occurring 497 times.

- **fbs**: 
  - There are two categories, and the most frequent one is "0" (indicating fasting blood sugar less than 120 mg/dl), which appears 872 times.

- **restecg**: 
  - Three unique results are present. The most common result is "1", appearing 513 times.

- **exang**: 
  - There are two unique values. The most frequent value is "0" (indicating no exercise-induced angina), which is observed 608 times.

- **slope**: 
  - Three unique slopes are present. The most frequent slope type is "1", which occurs 482 times.

- **ca**: 
  - There are five unique values for the number of major vessels colored by fluoroscopy, with "0" being the most frequent, occurring 578 times.

- **thal**: 
  - Four unique results are available. The most common type is "2" (indicating a reversible defect), observed 544 times.

- **target**: 
  - Two unique values indicate the presence or absence of heart disease. The value "1" (indicating the presence of heart disease) is the most frequent, observed in 526 entries.


## Exploratory Data Analysis

### Exploratory Data Analysis (EDA)

#### 1. Univariate Analysis:
   - Here, we'll focus on one feature at a time to understand its distribution and range.

#### 2. Bivariate Analysis:
   - In this step, we'll explore the relationship between each feature and the target variable.
   - This helps us figure out the importance and influence of each feature on the target outcome.

#### Objective:
   - With these two steps, we aim to gain insights into the individual characteristics of the data.
   - Additionally, we'll explore how each feature relates to our main goal: predicting the target variable.

#### Univiriate Analysis
##### Visualizing Numerical Data:
- **Histograms** are employed to gain insight into the distribution of each feature.
- This allows us to understand the central tendency, spread, and shape of the dataset's distribution.

##### Visualizing Categorical Data:
- **Bar plots** are utilized to visualize the frequency of each category.
- This provides a clear representation of the prominence of each category within the respective feature.

### Numerical Variables Univariate Analysis

![Capture2](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/d6406684-9894-48f1-8389-f5dcdb9adb17)


#### Age (age):
- The distribution is somewhat uniform, but there's a peak around the late 50s.
- The mean age is approximately 54.43 years with a standard deviation of 9.08 years.

#### Resting Blood Pressure (trestbps):
- The resting blood pressure for most individuals is concentrated around 120-145 mm Hg.
- The mean resting blood pressure is approximately 131.61 mm Hg with a standard deviation of 17.52 mm Hg.

#### Serum Cholesterol (chol):
- Most individuals have cholesterol levels between 200 and 300 mg/dl.
- The mean cholesterol level is around 246.00 mg/dl with a standard deviation of 51.59 mg/dl.

#### Maximum Heart Rate Achieved (thalach):
- The majority of the individuals achieve a heart rate between 140 and 170 bpm during a stress test.
- The mean heart rate achieved is approximately 149.11 bpm with a standard deviation of 23.01 bpm.

#### ST Depression Induced by Exercise (oldpeak):
- Most of the values are concentrated towards 0, indicating that many individuals did not experience significant ST depression during exercise.
- The mean ST depression value is 1.07 with a standard deviation of 1.18.

#### Overall:
- Upon reviewing the histograms of the numerical features and cross-referencing them with the provided feature descriptions, everything appears consistent and within expected ranges.
- There doesn't seem to be any noticeable noise or implausible values among the numerical variables.


### Categorical Variables Univariate Analysis

![Capture3](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/2daacf26-150c-4e8c-970e-7feae3e14d8f)

#### Gender (sex):
- The dataset is predominantly female, constituting a significant majority.

#### Type of Chest Pain (cp):
- The dataset shows varied chest pain types among patients. Type 0 (Typical angina) seems to be the most prevalent, but an exact distribution among the types can be inferred from the bar plots.

#### Fasting Blood Sugar (fbs):
- A significant majority of the patients have their fasting blood sugar level below 120 mg/dl, indicating that high blood sugar is not a common condition in this dataset.

#### Resting Electrocardiographic Results (restecg):
- The results show varied resting electrocardiographic outcomes, with certain types being more common than others. The exact distribution can be gauged from the plots.

#### Exercise-Induced Angina (exang):
- A majority of the patients do not experience exercise-induced angina, suggesting that it might not be a common symptom among the patients in this dataset.

#### Slope of the Peak Exercise ST Segment (slope):
- The dataset shows different slopes of the peak exercise ST segment. A specific type might be more common, and its distribution can be inferred from the bar plots.

#### Number of Major Vessels Colored by Fluoroscopy (ca):
- Most patients have fewer major vessels colored by fluoroscopy, with '0' being the most frequent.

#### Thalium Stress Test Result (thal):
- The dataset displays a variety of thalium stress test results. One particular type seems to be more prevalent, but the exact distribution can be seen in the plots.

#### Presence of Heart Disease (target):
- The dataset is nearly balanced in terms of heart disease presence, with about 51.3% having it and 48.7% not having it.


### Biviriate Analysis
#### Numerical Features vs Target Variables
![Capture4](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/68d19a69-a4c8-422b-81d0-50ea786f3a81)

##### Age (age):
- The distributions show a slight shift with patients having heart disease being a bit younger on average than those without. The mean age for patients without heart disease is higher.

##### Resting Blood Pressure (trestbps):
- Both categories display overlapping distributions in the KDE plot, with nearly identical mean values, indicating limited differentiating power for this feature.

##### Serum Cholesterol (chol):
- The distributions of cholesterol levels for both categories are quite close, but the mean cholesterol level for patients with heart disease is slightly lower.

##### Maximum Heart Rate Achieved (thalach):
- There's a noticeable difference in distributions. Patients with heart disease tend to achieve a higher maximum heart rate during stress tests compared to those without.

##### ST Depression (oldpeak):
- The ST depression induced by exercise relative to rest is notably lower for patients with heart disease. Their distribution peaks near zero, whereas the non-disease category has a wider spread.


#### Categorical vs Target
![Capture5](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/32e4a93d-d81a-4fc4-b063-9277ffc43b7c)

##### Number of Major Vessels (ca):
- The majority of patients with heart disease have fewer major vessels colored by fluoroscopy. As the number of colored vessels increases, the proportion of patients with heart disease tends to decrease. Especially, patients with 0 vessels colored have a higher proportion of heart disease presence.

##### Chest Pain Type (cp):
- Different types of chest pain present varied proportions of heart disease. Notably, types 1, 2, and 3 have a higher proportion of heart disease presence compared to type 0. This suggests the type of chest pain can be influential in predicting the disease.

##### Exercise Induced Angina (exang):
- Patients who did not experience exercise-induced angina (0) show a higher proportion of heart disease presence compared to those who did (1). This feature seems to have a significant impact on the target.

##### Fasting Blood Sugar (fbs):
- The distribution between those with fasting blood sugar > 120 mg/dl (1) and those without (0) is relatively similar, suggesting fbs might have limited impact on heart disease prediction.

##### Resting Electrocardiographic Results (restecg):
- Type 1 displays a higher proportion of heart disease presence, indicating that this feature might have some influence on the outcome.

##### Sex (sex):
- Females (1) exhibit a lower proportion of heart disease presence compared to males (0). This indicates gender as an influential factor in predicting heart disease.

##### Slope of the Peak Exercise ST Segment (slope):
- The slope type 2 has a notably higher proportion of heart disease presence, indicating its potential as a significant predictor.

##### Thalium Stress Test Result (thal):
- The reversible defect category (2) has a higher proportion of heart disease presence compared to the other categories, emphasizing its importance in prediction.

In summary, based on the visual representation:
- **Higher Impact on Target**: ca, cp, exang, sex, slope, and thal
- **Moderate Impact on Target**: restecg
- **Lower Impact on Target**: fbs


## Data Preprocessing
### Removing outliers
-![5](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/84d78b4f-4ed0-425b-8037-d2f91b297b2a)
### Categorical Encoding using pd.get_dummies()
-![6](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/d359d5bb-8514-4bc9-ae63-8bc18ec70737)

## Model Evaluation
- Data was splitted into training and testing sets using train_test_split() function from scikit-learn library.
- 80% of the data was used for training and 20% for testing.
- ![7](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/6a79708e-058d-48e8-9b49-c1512024b651)


### Model used 
#### Logistic Regression
- Purpose: Logistic Regression is a linear model used for binary classification tasks.
- Implementation: Implemented using the LogisticRegression class from scikit-learn.
- Parameters:
- **penalty**: Regularization term ('l1' or 'l2').
- **C**: Inverse of regularization strength.

- Usage
- ![8](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/d0ab870d-79b7-4f83-9839-8aa51dd39d42)

#### Decision Tree Classifier
- Purpose: Decision Trees are non-linear models used for classification tasks.
- Implementation: Implemented using the DecisionTreeClassifier class from scikit-learn.
- Parameters:
- **criterion**: Split criterion ('gini' or 'entropy').
- **max_depth**: Maximum depth of the tree.

- Usage
- ![9](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/17108cd2-5643-4bb1-b475-027973c8f737)

#### Random Forest Classifier
- Purpose: Random Forests are ensemble models combining multiple decision trees for improved performance.
- Implementation: Implemented using the RandomForestClassifier class from scikit-learn.
- Parameters:
- **n_estimators**: Number of trees in the forest.
- **max_depth**: Maximum depth of the trees.

- Usage
- ![10](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/8b9cfc3a-8c21-4fc3-84aa-41483d275c34)

#### K-Nearest Neighbors Classifier
- Purpose: K-Nearest Neighbors is a non-parametric method used for classification tasks.
- Implementation: Implemented using the KNeighborsClassifier class from scikit-learn.
- Parameters:
- **n_neighbors**: Number of neighbors to consider.
- **weights**: Weight function used in prediction.

- Usage
- ![11](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/9cf98c2c-5aaa-42df-9937-c1ce085819cc)

#### Support Vector Classifier (SVC)
- Purpose: Support Vector Classifier is a linear model used for binary classification tasks.
- Implementation: Implemented using the SVC class from scikit-learn.
- Parameters:
- **kernel**: Kernel function used ('linear', 'poly', 'rbf', 'sigmoid').
- **C**: Regularization parameter.

- Usage
- ![12](https://github.com/dzeble/Heart-Disease-Prediction-Group-2/assets/59400730/33ed35bf-2edc-4a20-bd6a-0ac6ce2bb3b8)

### Model Evaluation
A model is evaluated in terms of the samples correctly predicted as positive (True Positives), samples correctly predicted as negative (True Negatives), samples incorrectly predicted as positive (False Positives) and samples incorrectly predicted as negative (False Negatives).
Where TP = True Positives, TN = True Negatives, FP = False Positives,FN = False Negatives
  
#### Evaluation Metrics
- Accuracy: 
Accuracy measures the overall correctness of a model's predictions. It is calculated as the ratio of correctly predicted instances to the total instances.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

- Precision: 
Precision measures the proportion of true positive predictions out of all positive predictions made by the model. It quantifies how many of the instances predicted as positive are actually positive.

Precision = TP / (TP + FP)

- Recall: 
Recall measures the proportion of true positive predictions out of all actual positive instances. It quantifies how many of the actual positive instances were correctly identified by the model. 

Recall = TP / (TP + FN)

- F1 Score: 
F1 score is the harmonic mean of precision and recall, providing a balanced measure that considers both metrics. It strike a balance between precision and recall and can be used as a single metric to summarize a model's performance.

F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

#### Best Model Performace 
- A summary of the results achieved by the Support Vector Classifier (SVC) model for heart disease detection.

##### Evaluation results (SVC)
- Accuracy: 0.99
- Precision: 1.00
- Recall: 0.98
- F1 Score: 0.99

##### Interpretation
The SVC model demonstrates exceptional performance in accurately classifying instances of heart disease. With an accuracy of 0.99, it correctly predicts heart disease presence or absence in 99% of cases. The precision of 1.00 indicates that all positive predictions made by the model are indeed true positives, with no false positives. Similarly, the recall of 0.98 indicates that the model correctly identifies 98% of all true positive cases. The F1 score of 0.99 further confirms the model's balanced performance in terms of precision and recall.

#### Model Saving 
The trained SVC model was saved as a pickle using the Joblib library, which efficiently serializes Python objects to disk.

## Experiment Tracking 
The code uses MLflow for experiment tracking. It starts a new MLflow experiment for each model, logs metrics, and saves the best models.

### Overview
It is structured into several key components:

1. Model Training: Each model is trained using the train_model function, which logs the model name and trains the model on the provided training data.
2. Model Evaluation: The evaluate_model function is used to evaluate the performance of each model. It performs cross-validation, hyperparameter tuning (if a param_grid is provided), and logs metrics and parameters using MLflow. It also calculates and logs the training and testing scores.
3. Confusion Matrix Visualization: The plot_and_log_confusion_matrix function generates and logs a confusion matrix for each model, along with key metrics such as accuracy, precision, recall, and F1 score.
4. Model Comparison: The compare_best_models function compares the performance of all evaluated models, plots a comparison of their training and testing scores, and saves the best model.

### Workflow

The adopted workflow is as follows:

1. Data preprocessing 
2. Model training and evaluation using cross-validation
3. Hyperparameter tuning for each model using a predefined parameter grid
4. Logging model performance metrics (accuracy, precision, recall, F1-score) and confusion matrices using MLflow
5. Comparison of the best models based on test set accuracy
6. Saving the best model for future use

### Requirements

Python libraries include:

- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas
- mlflow

### Parameters
The code includes parameter grids for hyperparameter tuning of Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machines. These grids are used in the evaluate_model function to tune the models' hyperparameters. Voting Classifer is an ensemble of the above-mentioned models.

### Usage

1. Set up an MLflow tracking URI (e.g., `mlflow.set_tracking_uri("sqlite:///mlflow.db")`)
2. Ensure that the required data (X_train, X_test, y_train, y_test) is available
3. Run the code to train and evaluate the models
4. Review the logged metrics and artifacts in the MLflow UI
5. Inspect the comparison of the best models and the saved model file

### Artifacts
The project logs several artifacts using MLflow, including:
Model Artifacts: The best model for each evaluated model is saved as a pickle file.
Confusion Matrix Plots: A confusion matrix plot for each model is generated and logged.
Comparison Plot: A plot comparing the training and testing scores of all evaluated models is generated and displayed.