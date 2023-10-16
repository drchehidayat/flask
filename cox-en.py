#%% read the data and define the types
end = "OS"  # end="OS" or "DFS"

import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import copy
from pandas.api.types import CategoricalDtype

df = pd.read_csv('stroke_data.csv')
df

# Define type of features for categorical data
df["sex"] = df["sex"].astype("category")
df["dm"] = df["dm"].astype("category")
df["who"] = df["who"].astype("category")
df["gcs"] = df["gcs"].astype("category")
df["nihss"] = df["nihss"].astype("category")
df["mrs"] = df["mrs"].astype("category")
df["status"] = df["status"].astype("category")

print(df.dtypes)

# Split training set and test set
from sksurv.datasets import get_x_y
from sklearn.model_selection import train_test_split

X, y = get_x_y(df, attr_labels=['status', 'dur_month'], pos_label=1)
X0 = X
y0 = y
X, Xtest, y, ytest = train_test_split(X, y, test_size=0.3, random_state=1)
Xt = X
Xttest = Xtest
X

#%% COX-EN

#how the coefficients change for varying α
alphas = 10. ** np.linspace(-4, 4, 30)
coefficients = {}

cph = CoxPHSurvivalAnalysis()

for alpha in alphas:
    cph.set_params(alpha=alpha)
    cph.fit(X, y)
    key = round(alpha, 5)
    coefficients[key] = cph.coef_

coefficients = (pd.DataFrame
    .from_dict(coefficients)
    .rename_axis(index="feature", columns="alpha")
    .set_index(X.columns))

# ... (other code) ...

# Import the concordance_index_censored function
from sksurv.metrics import concordance_index_censored

# Calculate the concordance index
cindex = concordance_index_censored(ytest["status"], ytest["dur_month"], cph.predict(Xtest))
print(cindex)

# Import the brier_score function
from sksurv.metrics import brier_score

# ... (other code) ...
# Calculate predicted survival probabilities (preds)
survs = cph.predict_survival_function(Xtest)
preds = [fn(3) for fn in survs]  # Change the time as needed

times, coxscore3m = brier_score(y, ytest, preds, 3)
times, coxscore1y = brier_score(y, ytest, preds, 12)
times, coxscore3y = brier_score(y, ytest, preds, 36)

# ... (other code) ...
import joblib

# Save the Cox-EN model to a .pkl file
model_filename = "cox-en.pkl"
joblib.dump(cph, model_filename)
