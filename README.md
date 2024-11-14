# CodeClauseInternship_FraudDetection (Data Science)
Developed a machine learning model to detect fraudulent credit card transactions using Python, Pandas, and Scikit-learn. Learned to handle imbalanced datasets and improve fraud detection accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('creditcard.csv')
df.head()

( The dataset contains transactions made by credit cards in September 2013 by European cardholders.
  This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all 
  transactions.

  It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information 
  about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds 
  elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 
 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. )

# Preprocess the data
X = df.drop('Class', axis=1)
y = df['Class']

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

(The x-axis represents the two classes: 0 (non-fraudulent) and 1 (fraudulent).
 The y-axis represents the count of transactions in each class.
 This visualization helps you understand the imbalance in your data.
 Typically, you'll see far fewer fraudulent transactions compared to non-fraudulent ones.)

# Handle imbalanced dataset using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

(True Negatives (TN): Top-left cell (correctly predicted non-fraud).
False Positives (FP): Top-right cell (incorrectly predicted fraud).
False Negatives (FN): Bottom-left cell (incorrectly predicted non-fraud).
True Positives (TP): Bottom-right cell (correctly predicted fraud). )

# Visualize feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importance')
plt.show()

( The y-axis lists the top 10 features.
 The x-axis shows the importance scores assigned to these features by the model.
 Higher values indicate that the feature is more important for the model in making decisions.
 This helps in understanding which features are most influential in predicting fraudulent transactions. )

