# Breast_Cancer_Diagnosis_RFC
	I started writing my code by importing all the necessary libraries.
•	NumPy: For numerical computations, to calculate mean, accuracy and to store data(array)
•	Scikit-learn – datasets: To load breast cancer dataset.
•	Scikit-learn – train_test_split: To split dataset into training and testing sets.
•	Scikit-learn – RandomizedSearchCV: To perform hyperparameter tuning through randomized search.
•	Scikit-learn – cross_val_score: It is used for cross validation.
•	Scikit-learn – ensemble: To import the RandomForestClassifier.
•	Scikit-learn – metrics: To calculate precision, accuracy, recall scores on test dataset.
•	Matplotlip.pyplot: To create Visualizations of the confusion matrix and Decision tree.
•	Seaborn: To enhance visualization of confusion matrix by creating a heat map.
	This code is to diagnose breast cancer using Random Forest classifier.
	The breast cancer dataset is loaded from scikit-learn and then printed the feature names and the target labels, to know the structure of the dataset.
	Then I split the dataset into training and the testing dataset, by allocating 80% for training and testing with 20%.
	Firstly, I instantiated the Random Forest Classifier with default hyperparameters.
	Next,  I created a ‘parameters’ dictionary for hyperparameter values for tuning RFC. The parameters included are the number of trees (n_estimators), the maximum depth of each tree (max_depth), the minimum number of samples required to split a node (min_samples_split), and the minimum number of samples required at a leaf node (min_samples_leaf).
	Utilized RandomizedSearchCV to perform hyperparameter tuning on a RandomForestClassifier, exploring a range of hyperparameter combinations specified in the parameters dictionary, and prints the best hyperparameter values found through the search.
	Then, RandomForestClassifier is instantiated with the best hyperparameters obtained from the previous hyperparameter tuning using RandomizedSearchCV. The classifier is then trained on the training data (X_train, y_train). This ensures that the model is configured with the most optimal hyperparameters for making predictions on the specific dataset.
	The cross_val_score function is used to perform cross-validation (cv = 5) on the training data using the RandomForestClassifier with optimized hyperparameters. Thus, the resulting cross-validated accuracy scores are printed, and the mean accuracy across all folds is calculated and displayed. This step helps assess the model's generalization performance and stability by evaluating its accuracy on different subsets of the training data.
	The predict method is applied to the classifier to generate predictions (y_pred) on the test dataset (X_test). This step allows for assessing the model's performance on unseen data, as the predictions will be compared to the actual target values (y_test).
	The confusion_matrix function from scikit-learn is used to compute the confusion matrix for the predictions (y_pred) on the test set (y_test). The resulting confusion matrix is then "raveled" to obtain the four components: true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp).
	Finally printed the final performance metrics of a RandomForestClassifier on a test dataset, including specificity, sensitivity, accuracy, precision, recall, a classification report, and a confusion matrix.
	By utilizing Matplotlib and Seaborn created and displayed a heatmap of the confusion matrix, providing a visual representation of the classification results.
	At the end visualized the first decision tree from the Random Forest ensemble.
