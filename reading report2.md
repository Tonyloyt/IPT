
# Introduction To  Supervised Machine Learning : A Reading report

## Contents

    • What is supervised Machine Learning
    • Generalization, Overfitting, and Underfitting
    • Relation of Model complexity to dataset size
    • Supervised Machine Learning Algorithms
    • Uncertainty Estimates from Classifiers

## What is supervised Machine Learning

Supervised learning used in predictions of  outcome from certain given inputs, the model built from features/label pairs with human supervision on train the model with training set of data. Then the model is tested with unseen data so as to estimate how correct it is.
Types of supervised learning are classification and regression, Classification  deals with problems which produce binary output. It may be used to separate binary classes or multclasses   forexample classification  whether a person has diabetes or not (binary classification) and classification of iris species whether setosa, versicolor, or virginica (multclass classification).
Regression deals with problems which produce  continuous outputs. Forexample Extract stock sentiment from news headlines.

## Generalization, Overfitting, and Underfitting

• **Generalization:** - when the model is able to make accuracy predictions of on unseen data ,termed as generalization of the model from  training set to  test set.
    • **Underfitting:** - when the model doe not generalize from training set to test set, the model perform bad with during training set worse accuracy made during prediction  with both seen and unseen data.
    • **Overfitting:**-when the model tend to generalize well with training set but when comes to unseen data the model fail to genaralize. This show halfway performance with data.

## Relation of Model Complexity to Dataset Size

How the model complexity relate with dataset size?
The larger variety of data points your dataset contains, the more complex a model you can use without overfitting. Usually, collecting more data points will yield more variety, so larger datasets allow building more complex models.

## Supervised Machine Learning Algorithms

Supervised ML Algorithms to perform classification and regression problems, algorithms selection depends on the nature of the problem.
    • **k-Nearest Neighbors :** K-NN  ML algorithm which used for both classification and regression problems. The algorithm used to build the model with training data with the support of nearest neighbors. In classification K-NN used classifier to build the model with training set, known as KNeighborsClassifier while in regression K-NN use regressor called KNeighborsRegressor to train the model. The algorithm depends on the number of neighbors and the euclidean distance. K-NN is very easy to understand as well as to implement, the model is very fast with few data but tend to slow when the dataset is large.
    • **Linear Regression:** is ML algorithm which used fro regression  problems, the algorithm build the model from the linear equation,whenever the performance of linear Regression model is not good, One most commonly used alternatives to standard linear regression is ridge regression -use the coefficient which predict well training data with additional constraints, use alpha number to optimize the performance. Another is lasso which use lowest alpha  value to improve predictive performance.
    • **LogisticRegression :** the ML algorithm which perform classification task by using threshold value, the model have fixed  range of value unlike with linear regression , Logistic Regression range from 0 to 1 ,become more powerful and guarding against overfitting becomes increasingly important when considering more features also use reguralization parameter to improve it performance.
    • **Naive Bayes:** the ML algorithm for classification problem,naive Bayes models are so efficient is that they learn parameters by looking at each feature individually and collect simple per-class statistics from each feature.consist three types of classifier GaussianNB , BernoulliNB, and MultinomialNB . GaussianNB can be applied to any continuous data, while BernoulliNB assumes binary data and MultinomialNB assumes count data.
    • **Decision Trees:** is the algorithm which perform both classification and Regression problem. The model built based on decision of if-else conditions. Use DecisionTreeClassifier for classification  task and DecisionTreeRegressor for regression tasks
    • **Random forests:** used to build model for classification and regression tasks  with support of n_estimator which indicate the number of trees to combine on the model training. Use RandomForestClassifier for classification  task and RandomForestRegressor for regression tasks
    • **Support vector machine:** The algorithm used to perform both classification and regression tasks ,use classifier and regressor to buid models depend on the tasks. Its performance depend on the type of kernel used relatively to the nature of data. Work better with small/few  datasets
    •  **Neural Networks (Deep Learning):** The algorithm used to build both classification and regression models with the use of multilayer perceptrons. Main advantages is that they are able to capture information contained in large amounts of data and build incredibly complex models. Given enough computation time, data, and careful tuning of the parameters, neural
    networks often beat other machine learning algorithms

## Uncertainty Estimates from Classifiers

The ability of classifiers to provide uncertainty estimates of predictions , which can lead to negative outcome on real-life application which is very dangerous. In scikit-learn there two function which used to obtaion uncertainty estimates from classifiers,known as decision_function and predict_proba.

## Reference

[Introduction to Machine Learning with Python A Guide for Data Scientists chapter 1, 2 and 4: A.C. Muller & Sarah Guido (oreilly)](https://github.com/amueller/introduction_to_ml_with_python)
