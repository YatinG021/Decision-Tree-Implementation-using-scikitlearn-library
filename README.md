# Decision-Tree-Implementation-using-scikitlearn-library

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YATIN GOGIA

*INTERN ID*: CT04DH1053

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION* : 
Project Overview
This project addresses the critical problem of detecting breast cancer (benign or malignant tumors) using machine learning, specifically the Decision Tree algorithm. The methodology mirrors the step-by-step process and detailed explanations demonstrated in the referenced YouTube video. The project utilizes Python's Scikit-learn library for implementation and relies on the built-in Breast Cancer Wisconsin (Diagnostic) Dataset. The approach encompasses data exploration, feature selection, model training, evaluation, and visualization, providing interpretability and actionable results.

Step 1: Importing Libraries and Data
The project begins by importing essential Python libraries: pandas for data manipulation, numpy for numerical operations, matplotlib and seaborn for visualization, and Scikit-learn (sklearn) for machine learning tools.

The dataset is imported directly from Scikit-learn, offering a straightforward method to access a well-structured medical dataset. This dataset contains 569 instances, each representing a tumor characterized by 30 numerical features (such as radius, texture, perimeter, area, etc.). The target variable denotes whether the tumor is benign (non-cancerous) or malignant (cancerous).
Step 2: Exploring the Data
The video guides users to explore the loaded data. Understanding features' statistical distributions and relationships is crucial for effective modeling. Basic exploratory data analysis (EDA) includes checking the shape of the data, identifying missing values, and visualizing the target distribution.

Feature columns: 30

Samples: 569

Target: 1 indicates malignant; 0 indicates benign.

Step 3: Splitting Data Into Training and Test Sets
To evaluate model performance reliably, the dataset is split into training and testing sets using train_test_split from sklearn. A common split ratio is 70-80% for training and 20-30% for testing. This ensures the model is evaluated on unseen samples, guarding against overfitting and misleading accuracy scores.
