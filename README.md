# Hotel Booking Data Analysis

This project involves analyzing hotel booking data and making adr predictions based on various features, such as the length of stay, lead time, room type, number of guest and market segment ect.

### Data Preprocessing Steps

1. **Load Data**: 
   - The dataset is loaded from a CSV file.
   ```python
   df = pd.read_csv('/path/to/hotel_booking_data.csv')
Feature Engineering:

Lead Time is categorized into different bins based on quartiles to analyze booking trends.
ADR Binning: ADR (Average Daily Rate) is divided into bins using dynamic percentiles.
Meal Encoding: The meal column is encoded as a numerical variable using predefined mappings.
Country Quantiles: Country-specific ADR quantiles are mapped using an external CSV.
Market Segment and Distribution Channel: These features are binned and merged with the dataset for analysis.
Room Type ADR: A new column is added to categorize room types into ADR bins (Low, Medium, High) based on predefined ADR values.
Clean Up:

redundant columns such as country, market_segment, distribution_channel, and reserved_room_type are dropped from the dataset for efficiency.

Machine Learning Model
Model Loading:

A pre-trained Random Forest model is loaded using pickle which can be created by running the "Data_EDA_VIsualisations_Model_building.ipynb" notebook.
python

The dataset is aligned with the model's expected features by adding missing columns and ensuring they match the model's input such as "cancelled yes or no" or which room a client is assigned to because its future data.

ADR Prediction:

The model predicts the ADR (adr) for each entry in the dataset using the trained Random Forest model.

# Separate features and target variable
X = df.drop(columns=['adr'])
y = df['adr']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest - RMSE:", rmse_rf)
print("Random Forest - R²:", r2_rf)

The dataset is saved with the predicted ADR values for further analysis.

Conclusion
This analysis provides valuable insights into how different factors such as the length of stay, market segment, and distribution channel affect the ADR of hotel bookings. The trained model can be used to predict ADR for new bookings, helping to optimize pricing strategies.
