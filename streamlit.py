# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = 'Assignment2_Dataset.csv'
df = pd.read_csv(dataset, encoding='cp1252', parse_dates=['Date'], dayfirst=False, index_col='S/N')

# Data Preprocessing
# Remove duplicated rows
df.drop([3533, 2610, 8385], inplace=True)

df = df[(df['Rented_Bike_Count'] != -2) & (df['Rented_Bike_Count'] != 34180.0)]
df = df.dropna(subset=['Rented_Bike_Count'])
df['Hour'].interpolate(method='linear', inplace=True)  # Fill missing values in Hour using linear interpolation
df['Temperature'].interpolate(method='quadratic', inplace=True)
df['Temperature'] = df['Temperature'].round(decimals=1)  # Round off the interpolated values to one decimal place
df = df[(df['Temperature'] >= -41) & (df['Temperature'] <= 41)]

def calculate_humidity(temperature, dew_point):
    # formula
    numerator = np.exp((17.625 * dew_point) / (243.04 + dew_point))
    denominator = np.exp((17.625 * temperature) / (243.04 + temperature))
    relative_humidity = 100 * (numerator / denominator)

    # round to nearest whole number
    return np.round(relative_humidity)

# Impute rows with missing Humidity
df.loc[df['Humidity'].isnull(), 'Humidity'] = calculate_humidity(df['Temperature'], df['Dewpoint_Temp'])

df['Snowfall'].fillna('no_snowfall', inplace=True)
df = df.dropna(subset=['Hit_Sales'])

df['Open'] = df['Open'].replace('n', 'No')
df['Open'] = df['Open'].replace(['Y', 'yes', 'yes '], 'Yes')
df['Snowfall'] = df['Snowfall'].replace({'no_snowfall': 0, 'low': 1, 'medium': 2, 'heavy': 3, 'very heavy': 4})
df['Hit_Sales'] = df['Hit_Sales'].replace({'N': 0, 'Y': 1})
df['Open'] = df['Open'].replace({'No': 0, 'Yes': 1})

# Define the bin ranges and interpretive labels for each attribute
bin_ranges = {
    'Rented_Bike_Count': [-1, 0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500],
    'Temperature': [-30, -20, -10, 0, 10, 20, 30, 41],
    'Windspeed': [-1, 0.9, 2, 3, 5, 7.5],
    'Dewpoint_Temp': [-40, -20, 0, 20, 41],
    'Solar_Radiation': [-1, 0.01, 0.5, 1, 2, 4],
    'Rainfall': [-1, 5, 10, 20, 30, 36]
}

interpretive_labels = {
    'Rented_Bike_Count': ['No rentals', '1-100', '101-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500'],
    'Temperature': ['-30 to -20', '-20 to -10', '-10 to 0', '0 to 10', '10 to 20', '20 to 30', '30+'],
    'Windspeed': ['0-0.9', '0.9-2', '2-3', '3-5', '5+'],
    'Dewpoint_Temp': ['<-20', '-20 to 0', '0 to 20', '20+'],
    'Solar_Radiation': ['0', '0.01-0.5', '0.5-1', '1-2', '2+'],
    'Rainfall': ['No rain', '5-10', '10-20', '20-30', '30+']
}

# Perform binning and label assignment for each attribute
for attribute, bins in bin_ranges.items():
    df[f'{attribute}_Bins'] = pd.cut(df[attribute], bins=bins, labels=interpretive_labels[attribute])

def comfort_scale(row):
    score = 0
    if 20 <= row['Temperature'] <= 30:
        score += 1
    if 30 <= row['Humidity'] <= 60:
        score += 1
    if row['Windspeed'] <= 3:
        score += 1
    if row['Visibility'] <= 3:
        score += 1
    if row['Rainfall'] == 0:
        score += 1
    return score

df['Comfort_Scale'] = df.apply(comfort_scale, axis=1)

# Create new Columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

month_mapping = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

df['Month'] = df['Month'].map(month_mapping)
df['Day'] = df['Date'].dt.day_name()

df = df[df['Rented_Bike_Count'] > 0]

months = [['Dec', 'Jan', 'Feb'], ['Mar', 'Nov', 'Apr'], ['Oct', 'Aug', 'Sep', 'May', 'Jul'], 'Jun']

month_to_encoded = {}
encoded_month = 1

for month in months:
    if isinstance(month, list):
        for m in month:
            month_to_encoded[m] = encoded_month
        encoded_month += 1
    else:
        month_to_encoded[month] = encoded_month
        encoded_month += 1

df['Encoded_Month'] = df['Month'].map(month_to_encoded)

hours = [[4, 5], [0, 1, 2, 3], [6, 7], [10, 11], 9, [12, 13], [14, 15, 16], [20, 21, 22, 23], 8, [17, 18, 19]]

hour_to_encoded = {}
encoded_hour = 0

for hour in hours:
    if isinstance(hour, list):
        for h in hour:
            hour_to_encoded[h] = encoded_hour
        encoded_hour += 1
    else:
        hour_to_encoded[hour] = encoded_hour
        encoded_hour += 1

df['Encoded_Hour'] = df['Hour'].map(hour_to_encoded)

# Train the K-nearest neighbors model
SEED = 42

def train_knn_model():
    FEATURES = ['Temperature', 'Humidity', 'Solar_Radiation', 'Rainfall', 'Encoded_Hour', 'Encoded_Month', 'Comfort_Scale']
    BIKE = ['Rented_Bike_Count']

    Y = df[BIKE]
    X = df[FEATURES]

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=SEED)

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, Y_train)

    return knn_model

knn_model = train_knn_model()

FEATURES = ['Temperature', 'Humidity', 'Solar_Radiation', 'Rainfall', 'Encoded_Hour', 'Encoded_Month', 'Comfort_Scale']
HIT_SALES = ['Hit_Sales']       # classification
BIKE = ['Rented_Bike_Count']    # regression
Y = df[BIKE]               # set target
X = df[FEATURES]           # set x predictors

# Train the Random Forest model
def train_random_forest_model():
    RF_FEATURES = ['Temperature', 'Encoded_Hour', 'Humidity', 'Solar_Radiation', 'Encoded_Month', 'Rainfall', 'Comfort_Scale']
    HIT_SALES = ['Hit_Sales']

    Y = df[HIT_SALES]
    X = df[RF_FEATURES]

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=SEED)
    Y_train = Y_train.values.ravel()

    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    random_forest_model.fit(X_train, Y_train)

    return random_forest_model

random_forest_model = train_random_forest_model()

# Create Streamlit web app
st.title("Bike Rental and Sales Performance Prediction")

# User input section
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", -30.0, 41.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
solar_radiation = st.sidebar.slider("Solar Radiation (MJ/m^2)", 0.0, 5.0, 2.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 36, 10)
hour = st.sidebar.slider("Time (24-Hour format)", 0, 23, 12)
month = st.sidebar.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
windspeed = st.sidebar.radio("Is it windy?", ['Yes', 'No'])
visibility = st.sidebar.radio("Is weather expected to be clear?", ['Yes', 'No'])

# Calculate the comfort scale based on user input
comfort_scale = 0
if 20 <= temperature <= 30:
    comfort_scale += 1
if 30 <= humidity <= 60:
    comfort_scale += 1
if windspeed == 'No':
    comfort_scale += 1
if visibility == 'Yes':
    comfort_scale += 1
if rainfall == 0:
    comfort_scale += 1

# Encoding month and hour (replace this with the actual encoding logic)
encoded_hour = 0
encoded_month = 1

# Prepare data for prediction
input_data = {
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Solar_Radiation': [solar_radiation],
    'Rainfall': [rainfall],
    'Encoded_Hour': [encoded_hour],
    'Encoded_Month': [encoded_month],
    'Comfort_Scale': [comfort_scale]
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame(input_data)

# Predict bike rentals using the K-nearest neighbors model
predicted_rentals = knn_model.predict(input_df)

# Predict sales performance using the Random Forest model
predicted_sales = random_forest_model.predict(input_df)

# Display predictions
st.header("Predicted Results")
if len(predicted_rentals) > 0:
    st.write(f"You should expect around {round(predicted_rentals[0][0])} bicycle rentals.")
else:
    st.write("No prediction available.")
st.write(f"Hit sales are {'EXPECTED' if predicted_sales[0] == 1 else 'NOT EXPECTED'} to be achieved.")



