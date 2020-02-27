# Capstone Project 2 MLB Pitching Data - Using MLB Pitcher Data to Predict What Pitch is Coming Next

![github_header](https://user-images.githubusercontent.com/52009110/75495660-e10c1d00-5973-11ea-9acc-4b8b3133615c.jpg)

## Data Problem

What pitch will the pitcher throw next? Where in the zone will he throw it? Is the batter able to make an educated guess on what is coming next, or is he totally clueless and just reacting to what is coming? Sports teams are beginning to adopt data science and analytics to answer questions like these, and try to gain an advantage on the competition. If a model can reliably predict what pitch will be coming next, based on factors like score of the game, inning, the count of the at-bat, righty or lefty batter, righty or lefty pitcher. All of these factors can contribute to predicting what pitch is coming next, and we see in this project that they are important for this purpose.

## Who's Interested?

The client for this project would be the managers, pitching, and hitting coaches who are interested in these questions for decision-making, as well as the players themselves. Any leg up they can get on the opposition is a good thing and if there is any way to reliably predict pitches coming, that would be an advantage for the hitters. Hitting coaches would be interested in these models as well. They can coach their guys to jump on a fastball on a certain count or sit back on an off-speed pitch in certain situations. Models like the ones I built in this project could even be good for baserunners, once the hitters get on base. If they and the coaches had an idea of what pitches were coming, baserunners could steal on off-speed stuff as it will give them a better chance to swipe the base. 

## The Data Set

The data set used for this project was found ![here on Kaggle](https://www.kaggle.com/pschale/mlb-pitch-data-20152018) which featured a large data set containing 2.8 million entries, as well as a data set with all the player names and their corresponding Id's. I used the "pitches.csv" data set for this project, which included columns like:

* Ball count and strike count
* Pitcher/batter handedness (left or right)
* Pitcher/batter score when pitched
* Inning
* Where runners were at on base

All of these factors, including more that were created from the data, went into modeling this data and determining which columns were the most important to predicting the next pitch. 

## Modeling Philosophy

This ended up being a multi-class classification problem, mainly revolved around three classes based on the type of pitch thrown. Either a fastball, breaking ball, or off-speed pitch. Since this was a classification problem, classification algorithms were needed in order to correctly predict the coming pitch. Models included logistic regression, ridge classifier for the baseline modeling. For more in depth modeling, a competition was held between random forest, gradient boosting, ada boosting, and k-nearest neighbors. Gradient boosting won and a deep dive was performed into it, looking for feature importances, model accuracies, and classification reports.

Due to baseball's nature, and an overwhelming number of fastballs being thrown throughout the MLB, this ended up being an imbalanced multi-class classification problem. Using SMOTE oversampling, the data set was sampled and modeled again with the gradient boosting classifier in order to get better predictions on minority classes. It ended up serving its purpose and minority classes were able to be better predicted with the oversampling technique.

Neural networks were also used, MLP classifier from the Scikit-Learn library. Then a more in-depth neural network built using Keras and TensorFlow.

## Findings

* Pitcher side, strike count, ball count, batter side, and inning ended up being a few of the most important factors in determining what pitch was coming next
* For all the gradient boosting tests with oversampling or not, pitcher side was always the most important feature
* A few other important features were where runners were at on base (1st and 2nd base being the most important), whether it was the top of the inning or not, and whether the pitcher was behind in the count or not
