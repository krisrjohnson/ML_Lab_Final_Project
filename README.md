[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krisrjohnson/ML_Lab_Final_Project/blob/main/Final_Models.ipynb)

---
# Introduction

Final project for Machine Learning Lab course for University of San Francisco's Masters in Data Science Program.

---
## Project Goal

Can we accurately classify Bitcoin transactions into either Ransomware or non-ransomware categories?

---
## Data and EDA

Dataset from UCI Bitcoin Heist Ransomware. Brief Description from UCI:

	Entire Bitcoin transaction graph from 2009 January to 2018 December. Using a time interval of 24 hours, daily transactions on the network to form the Bitcoin graph. Filtered out network edges that transfer less than B0.3, as ransom amounts are rarely below this threshold.

- Large (for running locally) original dataset, of 2.9 million recrds
- Data extremely imbalanced
	- only 41k ransomWare
- A lot of heavy right tails
- Limit to 100 MB
	- downsample just the safe records to get there, after doing feature engineering
	- don't drop any ransomWare records

---
## Feature Engineering

- Convert multiclass labels into 0's or 1's
	- Less business value and more complexity in identifying specific 
- Have only numeric data and timestamps
	- Dropping timestamps
		- Possible data leakage, since ransomWare gathered only over certain years
	- Using timestamps to create cumulative sum columns per wallet, e.g. cumulative transactions
	- Using timestamps to create wallet age and 

I had to drop data to fit the assignment's size constraint. Only dropped majority class records to preserve as much signal as possible from the minority class. Ended up with about 600k total datapoints.

## Algorithms

Searching between good old Logistic Regression and Random Forest Classifier. Obvoiusly I expecte RandomForest to handily beat logistic regression, and that is ultimately the case, but it's a good exercise to see the difference between the two.

For both, using imblearn's sampling classes to help with the data imbalance. For logistic, using SMOTE to create synthetic datapoints from the minority ransomware class to upsample it's size to be even with or greater than the normal transaction class. For RandomForests, which take significantly longer to train as they're more complex than an analytical solution, using imblearn's `RandomUnderSampler()` to bring the majority class down in line with the minority ransomWare class.

For logistic, I further used StandardScaler to normalize the data, which is especially good as a lot of the columns have heavy, heavy right tails so it'll help squash those records closer to the rest, inhibiting the strength of these outliers. I also used PCA to further compress the data and to better be able to train in the direction of the most signal capture.  

For logistic itself, outside of switching to 'saga' solver, which is better for larger datasets, I mostly experiemented with the `C` hyperparameter, where 1/C represents regularization strength. So a larger `C` means less regularization. 

For RandomForests, we iterated over number of trees to fit as well as max depth.

## Metrics

We're primarily interested in finding the needle in the haystack and making sure we don't miss the needle more than being worried about pulling in a bunch of hay as well, so recall was the obvious choice of ultimate evaluation metric. However, fitting to recall alone leads to just pulling the entire haystack, so we trained on F-score weighted toward recall with a `beta=2`. 

## Results

RandomForests cleanly beat out LogisticRegression, although the default parameters performed the best in terms of Recall. However, I chose the tuned parameters since althogh their recall was slightly less the precision was significantly better to the point that it reduced the noise enough to be worth it.