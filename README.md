# Ensemble - Bagging

Most of the errors from a model’s learning are from three main factors: variance, noise, and bias. By using ensemble methods, we’re able to increase the stability of the final model and reduce the errors mentioned previously. By combining many models, we’re able to (mostly) reduce the variance, even when they are individually not great, as we won’t suffer from random errors from a single source.

Main principle behind ensemble modelling is to group weak learners together to form one strong learner.

BAGGING - Bootstrap Aggregation:

Bootstraping: - Statistica method where Creating N samples from Dataset of Size B. 
              - These samples are selected with replacement.
Aggregating:  - Multiple model fitting in parallel
              - For Regression Individual models are avegared
              - For Clasification Class O/p of each class is considered and each model is given a vote. The final voring is done through Hard Voting.

------------------------------------------

While calculating error, consider the error of the unused samples as well. This gives out of bag error {RMSE for Regression & ClassificationError for Classification}
While feature importance, consider GINI Index and Accumulate. Avg Informtion Gain across trees and accumulate

- Bagging Performance increases with more trees.

Disadv:
- Parallel trees are co-related & not of high impact if certain feature has high importance.


-------------------------------------------

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# load the data
url = "/home/debomit/Downloads/wine_data.xlsx"
dataframe = pd.read_excel(url)
arr = dataframe.values
X = arr[:, 1:14]
Y = arr[:, 0]

seed = 8
kfold = model_selection.KFold(n_splits = 3,
					random_state = seed)

# initialize the base classifier
base_cls = DecisionTreeClassifier()

# no. of base classifier
num_trees = 500

# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
						n_estimators = num_trees,
						random_state = seed)

results = model_selection.cross_val_score(model, X, Y, cv = kfold)
print("accuracy :")
print(results.mean())
