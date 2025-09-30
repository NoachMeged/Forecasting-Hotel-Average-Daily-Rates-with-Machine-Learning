Hotel Booking Data Analysis & ADR Prediction

This project focuses on analyzing hotel booking data and predicting the Average Daily Rate (ADR) for hotel reservations using machine learning. It provides insights into factors affecting hotel pricing and supports data-driven revenue management.

Project Overview

The analysis evaluates how various features influence ADR, including:

Stay Duration: Weekday and weekend nights.

Lead Time: Days between booking and arrival, categorized into quartiles.

Room Type: Categorized into ADR bins (Low, Medium, High).

Guest Composition: Number of adults, children, and babies.

Market Segment & Distribution Channel: Encoded and binned for predictive modeling.

Meal Plan: Numerically encoded.

Country: ADR quantiles per country.

Deposit Type: Encoded for model input.

The goal is to predict ADR for future bookings and provide actionable insights for pricing strategy optimization.

Data Preprocessing

Key preprocessing steps:

Feature Engineering

Lead Time binned by quartiles.

ADR dynamically binned using percentiles.

Meal plans and deposit types numerically encoded.

Country-specific ADR quantiles merged.

Market segment and distribution channel merged with ADR bins.

Room types categorized based on ADR averages.

Data Cleaning

Redundant columns (country, market_segment, distribution_channel, reserved_room_type) removed.

Machine Learning Model

A Random Forest Regressor is used for ADR prediction:

Model Training

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


Evaluation

RMSE and R² metrics assess model performance.

Predictions appended to the dataset for further analysis.

Feature Alignment

Ensure input data columns match the trained model’s expected features.

Insights & Applications

Identifies key drivers of ADR:

Booking lead time

Stay duration

Market segment and distribution channel

Room type and guest composition

Supports dynamic pricing and revenue optimization.

Can be integrated into an interactive Streamlit dashboard for real-time predictions.

Installation

Clone the repository:

git clone <repo-url>


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run Hotel_prediction_app_code.py

Contributing

Contributions are welcome for improvements in:

Data preprocessing

Feature engineering

Model performance

Visualization and dashboard features

Please submit pull requests with clear documentation.

License

This project is open-source and available for educational and research purposes.
