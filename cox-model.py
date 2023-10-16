import pandas as pd
from lifelines import CoxPHFitter

# Load the dataset
df = pd.read_csv('stroke_data.csv')

# Remove the 'id' column if it's present
if 'id' in df:
    df = df.drop(columns=['id'])

# Assuming you have already encoded the categorical variables as numbers, and 'status' as 0 or 1
X = df[['age', 'sex', 'dm', 'who', 'gcs', 'nihss', 'mrs']]
T = df['dur_month']
E = df['status']

# Create a Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='dur_month', event_col='status')

# Access the coefficients and their significance
print("Coefficients:")
print(cph.params_)

# Calculate Harrell's C-index for model performance
from lifelines.utils import concordance_index
c_index = concordance_index(T, -cph.predict_partial_hazard(X))
print(f"C-index: {c_index}")

# Save the Cox model to a file using joblib
import joblib
joblib.dump(cph, 'cox.pkl')
