import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('/Users/noachmeged/Downloads/hotel_booking_data.csv')

# Display the first few rows
df.head()

# Sample DataFrame (replace with your actual df)
dfweeks = pd.DataFrame({
    "stays_in_week_nights": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 30, 32, 33, 34, 35, 40, 41, 42, 50],
    "median_adr": [88.0, 89.1, 95.0, 99.0, 99.82, 89.13, 96.13, 88.4, 93.77, 99.45, 84.8, 79.175, 75.0, 88.9, 62.0, 74.49, 73.095, 79.59, 85.695, 57.8, 55.8, 72.005, 73.89, 98.33, 41.725, 0.26, 31.45, 42.11, 0.0, 0.0, 0.0, 27.145, 8.34, 110.5, 110.0]
})

# Find breakpoints dynamically (using percentiles of median ADR)
breakpoints = np.percentile(dfweeks["median_adr"], [20, 40, 60, 80])  # Adjust these percentiles if needed

# Define bin labels
bin_labels = [0, 1, 2, 3, 4]

# Apply binning based on median ADR values
dfweeks["adr_median_bins"] = pd.cut(dfweeks["median_adr"], bins=[-float("inf")] + breakpoints.tolist() + [float("inf")], labels=bin_labels)

# Group and summarize
adr_grouped_stats = dfweeks.groupby("adr_median_bins")["stays_in_week_nights"].agg(["count", "mean", "median"]).reset_index()

df['hotel'] = df['hotel'].map({'City Hotel': 0, 'Resort Hotel': 1})

import pandas as pd

# Zorg ervoor dat 'lead_time' numeriek is
df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')

# Verwijder rijen met NaN in 'lead_time'
df = df.dropna(subset=['lead_time'])

# Handmatig instellen van de kwartielen (uit je output)
q1, q2, q3 = 18.0, 69.0, 160.0

# Functie om lead_time in numerieke categorieën te verdelen
def bin_lead_time_q(days):
    if days <= q1:  
        return 0  # 0 - 18
    elif days <= q2:  
        return 1  # 18 - 69
    elif days <= q3:  
        return 2  # 69 - 160
    else:
        return 3  # 160+

# Pas de categorisatie toe op de bestaande kolom 'lead_time'
df['lead_time'] = df['lead_time'].apply(bin_lead_time_q)

# Define bin edges for arrival_date_day_of_month
day_bin_edges = [1, 8, 16, 23, 31]  # These are the bin edges you found earlier
day_bin_labels = [0, 1, 2, 3]  # Assign numeric labels to bins

# Create the day_bin column
df["day_bin"] = pd.cut(df["arrival_date_day_of_month"], bins=day_bin_edges, labels=day_bin_labels, include_lowest=True)

meal_mapping = {"SC": 0, "BB": 1, "HB": 2, "FB": 3}
df = df[df["meal"] != "Undefined"]  # Verwijder "Undefined"
df["meal"] = df["meal"].map(meal_mapping)

df_mapping = pd.read_csv('/Users/noachmeged/Downloads/country_adr_quantiles.csv')

# Convert mapping DataFrame into a dictionary
country_quantile_map = dict(zip(df_mapping['country'], df_mapping['country_adr_quantile']))

# Apply the mapping to create the 'country_adr_quantile' column
df['country_adr_quantile'] = df['country'].map(country_quantile_map)


# ADR data per market segment
data = {
    "market_segment": ["Online TA", "Direct", "Aviation", "Offline TA/TO", "Groups", "Corporate", "Undefined", "Complementary"],
    "adr": [117.186, 114.884, 100.142, 87.318, 79.432, 69.257, 15.000, 2.910]
}

# Convert dictionary to DataFrame
df_adr = pd.DataFrame(data)

# Create quantile bins (0 = lowest ADR, 3 = highest ADR)
df_adr["marketing_adr_bin"] = pd.qcut(df_adr["adr"], q=4, labels=[0, 1, 2, 3])

# Merge the bins into the main DataFrame based on the 'market_segment' column
df = df.merge(df_adr[['market_segment', 'marketing_adr_bin']], on='market_segment', how='left')

import pandas as pd


# Controleer of de kolom bestaat
if "distribution_channel" not in df.columns:
    raise KeyError("Kolom 'distribution_channel' ontbreekt in df!")

# Manually provided mean ADR values per distribution_channel
adr_mean_per_channel = {
    "Corporate": 69.097646,
    "Direct": 106.559617,
    "GDS": 120.554301,
    "TA/TO": 103.469652
}

# Convert dictionary to DataFrame
adr_df = pd.DataFrame(list(adr_mean_per_channel.items()), columns=["distribution_channel", "mean_adr"])

# Debug: Check inhoud van adr_df
print("ADR DataFrame:\n", adr_df)

# Check op unieke waarden en pas binning toe
if len(adr_df["mean_adr"].unique()) >= 4:
    adr_df["distribution_channel_adr_bin"] = pd.qcut(adr_df["mean_adr"], q=4, labels=[0, 1, 2, 3])
else:
    adr_df["distribution_channel_adr_bin"] = pd.cut(adr_df["mean_adr"], bins=4, labels=[0, 1, 2, 3])

# Zorg ervoor dat 'distribution_channel_adr_bin' categorisch blijft
adr_df["distribution_channel_adr_bin"] = adr_df["distribution_channel_adr_bin"].astype("category")

# Debug: Check of bins correct zijn aangemaakt
print("ADR DataFrame met bins:\n", adr_df)

# Merge bins back into the main DataFrame
df = df.merge(adr_df[["distribution_channel", "distribution_channel_adr_bin"]], on="distribution_channel", how="left")

# Debug: Check NaN-values
print("Aantal NaN in distribution_channel_adr_bin:", df["distribution_channel_adr_bin"].isnull().sum())

# **FIX**: Voeg de categorie "Unknown" toe voordat je fillna uitvoert
df["distribution_channel_adr_bin"] = df["distribution_channel_adr_bin"].astype("category")
df["distribution_channel_adr_bin"] = df["distribution_channel_adr_bin"].cat.add_categories(["Unknown"])
df["distribution_channel_adr_bin"] = df["distribution_channel_adr_bin"].fillna("Unknown")

# Define mapping
deposit_mapping = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}

# Apply encoding
df["deposit_type_encoded"] = df["deposit_type"].map(deposit_mapping)
df["deposit_type_encoded"].value_counts()

import pandas as pd

# Define mean ADR values for reserved room types
reserved_adr_means = {
    "A": 90.91, "B": 90.58, "C": 159.64, "D": 120.60, "E": 124.50, 
    "F": 167.37, "G": 175.90, "H": 188.22, "L": 124.67, "P": 0.00
}

# Define bin edges based on ADR values
adr_bins = [0, 100, 150, 200]  # Low (<100), Medium (100-150), High (>150)
bin_labels = ["Low", "Medium", "High"]

# Apply binning only to reserved_room_type
df["reserved_room_bin"] = pd.cut(
    df["reserved_room_type"].map(reserved_adr_means),
    bins=adr_bins, labels=bin_labels
)

# Convert to numerical labels for ML
df["reserved_room_bin"] = df["reserved_room_bin"].astype("category").cat.codes


import pandas as pd

# Bereken de kwartielgrenzen opnieuw op basis van de actuele data
quartiles = df['lead_time'].quantile([0.25, 0.50, 0.75])

# Functie om lead_time in categorieën te verdelen
def bin_lead_time_q(days):
    if days <= quartiles[0.25]:  
        return "short_term"  # 0 - Q1
    elif days <= quartiles[0.50]:  
        return "medium_term"  # Q1 - Q2
    elif days <= quartiles[0.75]:  
        return "long_term"  # Q2 - Q3
    else:
        return "very_long_term"  # Q3 - Max

# Voeg de gebinde kolom toe aan de dataset
df['lead_time'] = df['lead_time'].apply(bin_lead_time_q)

# Bekijk de verdeling van de nieuwe categorieën

df.drop(columns=["country","market_segment","distribution_channel","reserved_room_type","deposit_type"], inplace=True)

import pickle

# Define the path to your saved model
model_path = "/Users/noachmeged/Documents/random_forest_model.pkl"

# Load the model with pickle
with open(model_path, "rb") as file:
    random_forest_model = pickle.load(file)


import pandas as pd
import pickle

# 📌 Load the trained model
model_path = "/Users/noachmeged/Documents/random_forest_model.pkl"
with open(model_path, "rb") as file:
    random_forest_model = pickle.load(file)

# 📌 Get the features the model was trained on
model_features = random_forest_model.feature_names_in_

# 📌 Remove unexpected columns and fill missing ones with 0
df = df.reindex(columns=model_features, fill_value=0)  # Applied directly to df

# 📌 Predict ADR
df["adr"] = random_forest_model.predict(df)

# ✅ Print results
print(df[["adr"]].head())

# 📌 Save the updated dataset with ADR predictions
df.to_csv("dataset_with_predicted_adr.csv", index=False)
