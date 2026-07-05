# ==============================================================
# Project : Telecom Customer Churn Prediction using Machine Learning
# Author  : Ashi Saini
# ==============================================================

"""
Business Problem:
-----------------
Telecom companies lose customers due to competition.
This project predicts whether a customer will leave
the telecom company based on customer information.

Target Variable:
Churn
Yes -> Customer Leaves
No  -> Customer Stays
"""

# ==============================================================
# Import Libraries
# ==============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Display Settings
pd.set_option("display.max_columns", None)

# Graph Style
plt.style.use("ggplot")
sns.set_palette("Set2")

print("="*60)
print(" TELECOM CUSTOMER CHURN PREDICTION ")
print("="*60)

# ==============================================================
# Load Dataset
# ==============================================================

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset Loaded Successfully!")

# ==============================================================
# First Five Rows
# ==============================================================

print("\nFirst Five Rows\n")
print(df.head())

# ==============================================================
# Last Five Rows
# ==============================================================

print("\nLast Five Rows\n")
print(df.tail())

# ==============================================================
# Shape
# ==============================================================

print("\nDataset Shape")
print("----------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# ==============================================================
# Column Names
# ==============================================================

print("\nColumn Names")
print("----------------------")

for column in df.columns:
    print(column)

# ==============================================================
# Dataset Information
# ==============================================================

print("\nDataset Information")
print("----------------------")

print(df.info())

# ==============================================================
# Missing Values
# ==============================================================

print("\nMissing Values")
print("----------------------")

print(df.isnull().sum())

# ==============================================================
# Duplicate Rows
# ==============================================================

print("\nDuplicate Rows")
print("----------------------")

print(df.duplicated().sum())

# ==============================================================
# Statistical Summary
# ==============================================================

print("\nNumerical Summary")
print("----------------------")

print(df.describe())

# ==============================================================
# Categorical Summary
# ==============================================================

print("\nCategorical Summary")
print("----------------------")

print(df.describe(include="object"))

# ==============================================================
# Unique Values
# ==============================================================

print("\nUnique Values")
print("----------------------")

for column in df.columns:
    print(f"{column} : {df[column].nunique()}")

# ==============================================================
# Target Variable
# ==============================================================

print("\nTarget Variable Distribution")
print("----------------------")

print(df["Churn"].value_counts())

print("\nTarget Percentage")
print("----------------------")

print(round(df["Churn"].value_counts(normalize=True)*100,2))

# ==============================================================
# Data Cleaning
# ==============================================================

print("\nData Cleaning Started...")

# customerID is not useful for prediction
df.drop("customerID", axis=1, inplace=True)


# Convert blank spaces to NaN
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

# Convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove Duplicate Rows
#df.drop_duplicates(inplace=True)

print("\nData Cleaning Completed Successfully!")

# ==============================================================
# Dataset After Cleaning
# ==============================================================

print("\nDataset Shape After Cleaning")
print("----------------------")

print(df.shape)

print("\nMissing Values After Cleaning")
print("----------------------")

print(df.isnull().sum())

# ==============================================================
# Save Clean Dataset
# ==============================================================

df.to_csv("../data/cleaned_telco_churn.csv", index=False)

print("\nClean Dataset Saved Successfully!")

print("\nPart 1 Completed Successfully!")


# ==============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# Create images folder
import os

os.makedirs("../images", exist_ok=True)


# ==============================================================
# Function to Plot Countplots
# ==============================================================

def plot_countplot(column, title, filename):

    plt.figure(figsize=(8,5))

    ax = sns.countplot(
        data=df,
        x=column,
        hue="Churn",
        palette="Set2"
    )

    # Add Count Labels
    for container in ax.containers:
        ax.bar_label(container)

    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel(column)
    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(f"../images/{filename}", dpi=300)

    plt.close()

    print("="*60)
    print(title)
    print("="*60)

    churn_table = pd.crosstab(
        df[column],
        df["Churn"],
        normalize="index"
    ) * 100

    print(churn_table.round(2))
    print()


# ==============================================================
# Function to Plot Histograms
# ==============================================================

def plot_histogram(column, title, filename):

    plt.figure(figsize=(8,5))

    sns.histplot(
        data=df,
        x=column,
        bins=30,
        kde=True
    )

    plt.title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()

    plt.savefig(f"../images/{filename}", dpi=300)

    plt.close()


# ==============================================================
# Function to Plot Boxplots
# ==============================================================

def plot_boxplot(column, title, filename):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="Churn",
        y=column
    )

    plt.title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()

    plt.savefig(f"../images/{filename}", dpi=300)

    plt.close()


# ==============================================================
# Countplots
# ==============================================================

plot_countplot(
    "gender",
    "Gender vs Churn",
    "gender_vs_churn.png"
)

plot_countplot(
    "SeniorCitizen",
    "Senior Citizen vs Churn",
    "seniorcitizen_vs_churn.png"
)

plot_countplot(
    "Partner",
    "Partner vs Churn",
    "partner_vs_churn.png"
)

plot_countplot(
    "Dependents",
    "Dependents vs Churn",
    "dependents_vs_churn.png"
)

plot_countplot(
    "Contract",
    "Contract vs Churn",
    "contract_vs_churn.png"
)

plot_countplot(
    "InternetService",
    "Internet Service vs Churn",
    "internetservice_vs_churn.png"
)

plot_countplot(
    "PaymentMethod",
    "Payment Method vs Churn",
    "paymentmethod_vs_churn.png"
)


# ==============================================================
# Histograms
# ==============================================================

plot_histogram(
    "tenure",
    "Tenure Distribution",
    "tenure_distribution.png"
)

plot_histogram(
    "MonthlyCharges",
    "Monthly Charges Distribution",
    "monthlycharges_distribution.png"
)

plot_histogram(
    "TotalCharges",
    "Total Charges Distribution",
    "totalcharges_distribution.png"
)


# ==============================================================
# Boxplots
# ==============================================================

plot_boxplot(
    "MonthlyCharges",
    "Monthly Charges vs Churn",
    "monthlycharges_boxplot.png"
)

plot_boxplot(
    "TotalCharges",
    "Total Charges vs Churn",
    "totalcharges_boxplot.png"
)

plot_boxplot(
    "tenure",
    "Tenure vs Churn",
    "tenure_boxplot.png"
)

print("\nEDA Completed Successfully!")

# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================
# Create Copy
# ==============================================================

df_ml = df.copy()

# ==============================================================
# Drop Customer ID (Not useful for prediction)
# ==============================================================



# ==============================================================
# Convert TotalCharges to Numeric
# ==============================================================

df_ml["TotalCharges"] = pd.to_numeric(
    df_ml["TotalCharges"],
    errors="coerce"
)

# Fill Missing Values
df_ml["TotalCharges"].fillna(
    df_ml["TotalCharges"].median(),
    inplace=True
)



# ==============================================================
# Label Encoding
# ==============================================================

from sklearn.preprocessing import LabelEncoder

df_ml = df.copy()

label_encoder = LabelEncoder()

# Encode every non-numeric column
for column in df_ml.columns:

    if not pd.api.types.is_numeric_dtype(df_ml[column]):
        df_ml[column] = label_encoder.fit_transform(df_ml[column])

print("\nLabel Encoding Completed!")

print(df_ml.head())

print("\nData Types After Encoding")
print(df_ml.dtypes)

print("\nFeature Order:")
print(df_ml.columns.tolist())



# ==============================================================
# Features and Target
# ==============================================================

X = df_ml.drop("Churn", axis=1)
y = df_ml["Churn"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# ==============================================================
# Train Test Split
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

# ==============================================================
# Check Data Types Before Scaling
# ==============================================================

print("\nChecking Data Types Before Scaling...\n")
print(X_train.dtypes)

# ==============================================================
# Feature Scaling
# ==============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("\nScaled Training Shape :", X_train_scaled.shape)
print("Scaled Testing Shape :", X_test_scaled.shape)

print("\nPart 2 Completed Successfully!")


# ==============================================================
# MACHINE LEARNING MODELS
# ==============================================================

print("\n" + "="*60)
print("TRAINING MACHINE LEARNING MODELS")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Dictionary of Models
models = {

    "Logistic Regression": LogisticRegression(random_state=42),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes": GaussianNB()

}

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    trained_models[name] = model

print("\nAll Models Trained Successfully!")


# ==============================================================
# MODEL EVALUATION
# ==============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

results = []

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

for name, model in trained_models.items():

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"\n{name}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


# ==============================================================
# MODEL COMPARISON TABLE
# ==============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

print(results_df)

results_df.to_csv(
    "../data/model_comparison.csv",
    index=False
)

# ==============================================================
# MODEL ACCURACY COMPARISON GRAPH
# ==============================================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy"],
)

# Add Accuracy Labels
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.title(
    "Machine Learning Model Accuracy Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Models", fontsize=12)

plt.ylabel("Accuracy", fontsize=12)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "../images/model_accuracy_comparison.png",
    dpi=300
)

plt.close()

print("\nAccuracy Comparison Graph Saved Successfully!")


# ==============================================================
# BEST MODEL
# ==============================================================

best_model_info = results_df.iloc[0]

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)

print(f"Model Name : {best_model_info['Model']}")
print(f"Accuracy   : {best_model_info['Accuracy']:.4f}")

# ==============================================================
# SELECT BEST MODEL OBJECT
# ==============================================================


best_model_name = best_model_info["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model Loaded Successfully!")

# ==============================================================
# CONFUSION MATRIX
# ==============================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

plt.figure(figsize=(6,6))

disp.plot(cmap="Blues")

plt.title(
    f"Confusion Matrix ({best_model_name})",
    fontsize=15,
    fontweight="bold"
)

plt.savefig(
    "../images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion Matrix Saved Successfully!")

# ==============================================================
# ROC CURVE
# ==============================================================

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_prob = best_model.predict_proba(X_test_scaled)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=3,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    f"ROC Curve ({best_model_name})",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../images/roc_curve.png",
    dpi=300
)

plt.close()

print("ROC Curve Saved Successfully!")


# ==============================================================
# FEATURE IMPORTANCE
# ==============================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Random Forest Model
rf_model = trained_models["Random Forest"]

# Feature Importance
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": importance

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# Plot

plt.figure(figsize=(10,8))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "Feature Importance (Random Forest)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig(
    "../images/feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature Importance Graph Saved Successfully!")

# ==============================================================
# SAVE MODEL
# ==============================================================

import joblib
import os

os.makedirs("../models", exist_ok=True)

joblib.dump(
    best_model,
    "../models/best_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("\nBest Model Saved Successfully!")

print("Scaler Saved Successfully!")


# ==============================================================
# Logistic Regression
# ==============================================================

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=42)

log_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# ==============================================================
# Prediction
# ==============================================================

y_pred = log_model.predict(X_test_scaled)

print("\nPrediction Completed!")

# ==============================================================
# Model Accuracy
# ==============================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100,2),"%")

# ==============================================================
# Classification Report
# ==============================================================

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# ==============================================================
# CONFUSION MATRIX
# ==============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"Confusion Matrix ({best_model_name})")

plt.tight_layout()

plt.savefig("../images/confusion_matrix.png", dpi=300)

plt.close()

print("Confusion Matrix Saved Successfully!")
# ==============================================================
# PROJECT COMPLETED
# ==============================================================

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("""
Project Summary

✔ Data Cleaning
✔ Exploratory Data Analysis
✔ Feature Engineering
✔ Train-Test Split
✔ Feature Scaling
✔ Logistic Regression
✔ Decision Tree
✔ Random Forest
✔ KNN
✔ Naive Bayes
✔ Model Comparison
✔ Accuracy Graph
✔ Best Model Selection
✔ Confusion Matrix
✔ ROC Curve
✔ Feature Importance
✔ Model Saved

All outputs are saved successfully.
""")

