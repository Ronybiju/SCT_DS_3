# SCT_DS_3
🧠 Bank Marketing Campaign Prediction using Decision Tree
This project uses the Bank Marketing dataset from the UCI Machine Learning Repository to predict whether a client will subscribe to a term deposit based on demographic and behavioral data. A Decision Tree Classifier is built, evaluated, and visualized, making this a complete end-to-end machine learning project.

📁 Dataset
File Name	Description
bank-additional-full.csv	Full dataset (41,188 records, 20 input features) used for prediction and analysis

Source: UCI Bank Marketing Dataset

Target Variable: y (whether the client subscribed to a term deposit: yes or no)

📌 Objectives
Load and preprocess real-world marketing data

Analyze patterns between demographic, contact, and financial behavior

Encode categorical features

Train a Decision Tree Classifier

Evaluate using metrics like accuracy, confusion matrix, and classification report

Visualize the decision tree and most important features

🛠️ Technologies Used
Python 🐍

Pandas & NumPy 📊

Matplotlib & Seaborn 📈

Scikit-learn (DecisionTreeClassifier) 🤖

Jupyter Notebook 📒

🔍 Steps Performed
✅ 1. Data Loading
python
Copy
Edit
df = pd.read_csv('bank-additional-full.csv', sep=';')
✅ 2. Data Cleaning & Exploration
Verified shape, types, and null values

Cleaned column names using .str.strip()

Checked class imbalance in target variable y

✅ 3. Feature Encoding
python
Copy
Edit
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
✅ 4. Train-Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
✅ 5. Model Training
python
Copy
Edit
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
✅ 6. Model Evaluation
python
Copy
Edit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
✅ 7. Tree & Feature Importance Visualization
🌳 Plot Decision Tree
python
Copy
Edit
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
🔥 Feature Importance
python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', title="Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.show()
📊 Results
Validation Accuracy: ~88%

Most influential features:

duration

emp.var.rate

euribor3m

poutcome

🚀 Future Enhancements
Try other models (Random Forest, XGBoost, LightGBM)

Handle class imbalance using SMOTE or class weights

Use GridSearchCV for hyperparameter tuning

Build a dashboard using Streamlit for business users
