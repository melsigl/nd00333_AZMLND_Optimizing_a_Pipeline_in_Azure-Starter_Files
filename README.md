# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset provided by UCI Machine Learning Repostory contains information regarding direct bank marketing campaigns. More specifically, it contains information about the age, job, day of contact, contact type such as telephone and how long a banking clerk had contact with (potential) customers. Based on those information we seek to predict if a client subscribet to a term deposit after contact.

The best performing model was a VotingEnsemble obtained by AutoML, which outperformed logistic regression by 0.0072 accuracy points.

## Scikit-learn Pipeline
The pipeline architecture consisted of downloading the data followed by cleaning the data. More specifically, one-hot encoding of categorical attributes, remove data records with missing values, and splitting data into training and test data according to a splitting ratio of 0.8:0.2. As the dataset is a binary classification problem, logistic regression will serve as a fairly simple but powerful baseline model.

Using HyperDrive, a hyperparameter tuning framework originally proposed by [Rasley et al. at Middleware ’17 conference](https://dl.acm.org/doi/10.1145/3135974.3135994), we tune exactly two hyperparameter, namely the regularization strength and the maximum number of iterations it will take to converge.

The first hyperparameter, regularization strength, is a floating number. Thus, a random parameter sampler using a uniform distribution is  used for this hyperparameter. In the case of maximun number of iterations, which inherently is an integer value, the random sampler selects from a predefined number of iterations. More specifically, prime numbers between 11 and 59. 

When comparing grid search and random search, the latter finds good hyperparameters more quickly. Therefore, in this experiment i chose a random parameter sampler.

Furthermore, as an early stopping stragety bandit policy was selected. Such a bandit policy is defined by three parameters: slack factor, evaluation interval, and delay evaluation. The *slack factor* specifies a ratio in which it is allowed to differ from the best performing model so far in an experiment run. The *evaluation interval* specifies the frequency in which this particular policy is applied, and *delay evaluation* determines the number of dormant intervals before this policy is applied again. Therefore, *delay evaluation* helps to avoid premature termination of training runs. With an slack factor of 0.1, evaluation interval of one, and a delay evaluation of 3 we seek to find a good logistic regression with this early stopping strategy.

Based on this described pipeline, logistic regression with regularization strength of 0.5594 and accuracy of 0.9099 was the best performing model.

## AutoML
Additionally, and complementary to the described Scikit-learn pipeline, I was wondering how an AutoML pipeline would perform, what kind of algorithm is the best, and if the best model would actually outperform a hand-selected model. Based on this curiousity, I configured a AutoML pipeline that handles dataset preparation tasks, followed by model selection and training fully automatically. In particular, AutoML generates features, imputes missing values, determines the status of class imbalances without human interference, and splitting data in cross-validation sets according to a user-defined split (set to 3 in this particular experiment). AutoML is fully cabable in selecting a vast range of diverse Machine Learning models. Additionally, it trains and validates these models with different hyperparameters and reporting their individual performance.

Not only is it possible to train multiple models automatically with AutoML, it furthermore reports the best performing one. In particular, given the bank marketing campaign dataset, AutoML reported VotingEnsemble as the best performing model with a final accuracy of 0.9171. This model contained 25 estimators in total, each of it has been trained in previous AutoML iterations. Therefore, AutoML combined already trained models using a soft voting. Interesstingly, attributes "duration", "nr.employed", "cons.conf.idx", and "euribor3m" have been voted to be the most important features in this experiment.


## Pipeline comparison
Comparing the best performing models accuracy-wise from those two described pipelines, VotingEnsemble from AutoML performed best. Specifically, logistic regression reached accuracy of 0.9099 whereas VotingEnsemble reached the higher accuracy of 0.9171. This increase in performance is easily explained in combining multiple _weaker_ estimators such as to gain a good prediction. 

## Future work
Although all trained models reached good performance, with VotingEnsemble reaching the highest accuracy of 0.9171 in my experiment, there is still room for improvement. For instance, AutoML reported class imbalances. This in itself imposes serious problems as one class is favoured due to a skewed and biased class distribution of the labels. In order to aleviate this problem, I plan to oversample the minority class by using the common method SMOTE.  

Additionally, I purposely limited the number of iterations HyperDrive performed due to time constraints. Likewise, AutoML only trained models for a duration of 30 minutes. With more time and maybe even access to GPUs, I plan to extend the number of iterations of HyperDrive and simultaneously extend the number and type of machine learning models AutoML trains. For instance, with GPUs available, I could test how simple neural networks perform on this particular dataset.
