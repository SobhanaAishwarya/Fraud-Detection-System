Project Name: Fraud Detection System 

This project focuses on detecting fraudulent credit card transactions using classification and anomaly detection techniques. It aims to identify rare fraud cases from highly imbalanced datasets.

*Features:Exploratory data analysis,SMOTE for class imbalance handling

*Classification models: Logistic Regression, Decision Tree, Neural Network

*Anomaly detection: Isolation Forest, Autoencoder
ROC-AUC, confusion matrix, and performance comparison

*Technologies Used:Python,Scikit-learn,Pandas, NumPy,Seaborn, Matplotlib,imbalanced-learn (SMOTE),TensorFlow/Keras (for Autoencoder)

*Dataset:Credit card transactions dataset with PCA-transformed features, amount, time, and a target label indicating fraud or not.

*Project Workflow:Data loading and preprocessing,Class balancing with SMOTE,Training multiple models,Evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC

*Results:
The models are evaluated and compared using ROC curves and metrics. Anomaly detection models show promising results due to the rarity of fraud cases.
