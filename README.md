# Decision-Tree-Implementation-using-scikitlearn-library

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YATIN GOGIA

*INTERN ID*: CT04DH1053

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION* :
# Breast Cancer Detection Using Decision Tree

## Project Overview

This project tackles the critical problem of detecting breast cancer (benign or malignant tumors) using machine learning, specifically the **Decision Tree algorithm**. The methodology follows a step-by-step process inspired by an instructional YouTube video and is implemented using Python’s Scikit-learn library. The model utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset, and the workflow includes data exploration, feature selection, model training, evaluation, and visualization, providing both accuracy and interpretability.

---

## Table of Contents

- [Importing Libraries and Data](#importing-libraries-and-data)  
- [Exploring the Data](#exploring-the-data)  
- [Data Splitting](#data-splitting)  
- [Building the Decision Tree Model](#building-the-decision-tree-model)  
- [Evaluation and Predictions](#evaluation-and-predictions)  
- [Feature Importance Interpretation](#feature-importance-interpretation)  
- [Decision Tree Visualization](#decision-tree-visualization)  
- [Model Pruning and Overfitting Control](#model-pruning-and-overfitting-control)  
- [Key Takeaways](#key-takeaways)  

---

## Importing Libraries and Data

The project begins by importing the primary Python libraries for data handling, visualization, and machine learning:

- `pandas` for data manipulation  
- `numpy` for numerical computations  
- `matplotlib` and `seaborn` for data visualization  
- `sklearn` for machine learning tools  

The Breast Cancer Wisconsin (Diagnostic) Dataset is loaded directly from `sklearn.datasets`. This dataset contains 569 samples, each with 30 numerical features describing tumor characteristics, and a binary target indicating whether a tumor is benign or malignant.

---

## Exploring the Data

Preliminary exploratory data analysis (EDA) is conducted to understand the dataset:

- Check data shape and feature columns  
- Identify if any missing values exist  
- Visualize the distribution of target classes  

Key details:
- **Features:** 30 numerical features (e.g., radius, texture, perimeter)  
- **Samples:** 569 tumor instances  
- **Target:** 1 = malignant, 0 = benign  

---

## Data Splitting

To evaluate model performance properly, the data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

This split (typically 80% training, 20% testing) prevents overfitting and validates the model on unseen data.

---

## Building the Decision Tree Model

The core of this project is a Decision Tree Classifier, built and trained as follows:

- Import the classifier:  
from sklearn.tree import DecisionTreeClassifier
- Initialize the model (with tuned depth to prevent overfitting):  
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
- Fit the classifier on training data:  
clf.fit(X_train, y_train)

The model's depth (`max_depth`) controls complexity; deeper trees can overfit, shallower trees might underfit.

---

## Evaluation and Predictions

After training, predictions are made on the test set:

- `predict(X_test)` outputs tumor class predictions  
- `predict_proba(X_test)` provides probabilities for each class  

Performance metrics calculated include accuracy, confusion matrix, precision, recall, and a classification report. These metrics ensure the model's effectiveness and reliability—especially important in a medical context.

---

## Feature Importance Interpretation

Decision Trees provide inherent interpretability by calculating feature importances. This shows which tumor features most influenced the classification. Visualizing feature importances with a bar chart helps highlight the most critical characteristics, such as “worst perimeter” or “mean area,” aiding clinical insight.

---

## Decision Tree Visualization

Visualization of the trained Decision Tree exposes the logic behind predictions:

- Displays decision nodes with feature splits, thresholds, and class distributions  
- Reveals the sequential rules leading to the final diagnosis  
- Helps validate model decisions and debug issues  

Tools like `sklearn.tree.plot_tree` or exporting to Graphviz can generate these visuals.

---

## Model Pruning and Overfitting Control

Overfitting is addressed by limiting tree depth (`max_depth`) and applying pruning techniques (`ccp_alpha` for cost-complexity pruning). These steps reduce model complexity, improve generalization, and typically achieve better accuracy on the unseen test data.

---

## Key Takeaways

- The Decision Tree algorithm accurately classifies malignant vs. benign tumors with a solid performance on unseen data.  
- Visualization and feature importance provide transparency and trust in the model for medical practitioners.  
- Hyperparameter tuning and pruning strike the balance between underfitting and overfitting.  
- This end-to-end project exemplifies reliable, interpretable AI for healthcare diagnostic support.

---

## Summary

This project demonstrates a comprehensive machine learning workflow for breast cancer detection using Decision Trees, from data processing through model training, evaluation, feature interpretation, and visualization. The result is a transparent and actionable diagnostic tool that supports medical decision-making.

---

## OUTPUTS
## Output Visualizations

### 1. Feature Importance Bar Chart
Displays the top features influencing the Decision Tree's classification, highlighting "mean concave points" as the most significant for breast cancer detection.
[<img width="676" height="596" alt="Image" src="https://github.com/user-attachments/assets/d55d9c75-8db3-407c-879f-ceaeaa38700f" />]

### 2. Decision Tree Structure Diagram
Shows the trained Decision Tree's decision paths and feature splits, making the model's classification logic transparent and interpretable.
<img width="1957" height="1560" alt="Image" src="https://github.com/user-attachments/assets/623cc116-0783-40ca-b571-a3b2de81974d" />

### 3. Test Set Prediction Dot Plot
Visualizes the distribution of prediction results across the test dataset, helping identify patterns or potential outliers in model performance.









