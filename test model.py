import pandas as pd
from lifelines import CoxPHFitter
import joblib  # Import the joblib module


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

# Define the specific time points you're interested in (e.g., 3 months, 1 year, and 3 years)
time_points = [3, 12, 36]  # Time points in months

# Calculate the cumulative hazard at each time point
cumulative_hazard = cph.predict_cumulative_hazard(X)

# Calculate the hazard rate at each time point using the cumulative hazard
for time_point in time_points:
    hazard_rate = cumulative_hazard.iloc[time_point].div(cumulative_hazard.iloc[time_point].max())
    print(f"Hazard Rate at {time_point} months:")
    print(hazard_rate)

# Save the Cox model to a file using joblib
joblib.dump(cph, 'test.pkl')