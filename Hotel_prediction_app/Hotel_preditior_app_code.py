import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the input data based on the uploaded dataset and mappings
def preprocess_data(df, df_mapping, df_adr, adr_df):
    try:
        st.write("Processing your uploaded dataset for ADR prediction.")
        
        # Mapping hotel types
        df['hotel'] = df['hotel'].map({'City Hotel': 0, 'Resort Hotel': 1})
        
        # Convert lead_time to numeric and handle missing values
        df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')
        df = df.dropna(subset=['lead_time'])
        
        # Categorize lead_time into bins
        quartiles = df['lead_time'].quantile([0.25, 0.50, 0.75])
        def bin_lead_time_q(days):
            if days <= quartiles[0.25]: return 0  # Short-term
            elif days <= quartiles[0.50]: return 1  # Medium-term
            elif days <= quartiles[0.75]: return 2  # Long-term
            else: return 3  # Very long-term

        df['lead_time'] = df['lead_time'].apply(bin_lead_time_q)

        # Encode meal types
        meal_mapping = {"SC": 0, "BB": 1, "HB": 2, "FB": 3}
        df = df[df["meal"] != "Undefined"]
        df["meal"] = df["meal"].map(meal_mapping)
        
        # Encode months
        month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                         "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
        df["arrival_date_month"] = df["arrival_date_month"].map(month_mapping)
        
        # Merge country ADR quantile mapping
        country_quantile_map = dict(zip(df_mapping['country'], df_mapping['country_adr_quantile']))
        df['country_adr_quantile'] = df['country'].map(country_quantile_map)
        
        # Merge ADR bins for marketing segment and distribution channel
        df = df.merge(df_adr[['market_segment', 'marketing_adr_bin']], on='market_segment', how='left')
        df = df.merge(adr_df[['distribution_channel', 'distribution_channel_adr_bin']], on='distribution_channel', how='left')
        df["distribution_channel_adr_bin"] = df["distribution_channel_adr_bin"].replace("Unknown", -1).astype(float)
        
        # Encode deposit type
        deposit_mapping = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
        df["deposit_type_encoded"] = df["deposit_type"].map(deposit_mapping)
        
        # Bin reserved room types into categories based on ADR
        reserved_adr_means = {"A": 90.91, "B": 90.58, "C": 159.64, "D": 120.60, "E": 124.50, 
                               "F": 167.37, "G": 175.90, "H": 188.22, "L": 124.67, "P": 0.00}
        adr_bins = [0, 100, 150, 200]
        bin_labels = ["Low", "Medium", "High"]
        df["reserved_room_bin"] = pd.cut(df["reserved_room_type"].map(reserved_adr_means),
                                          bins=adr_bins, labels=bin_labels).astype("category").cat.codes
        
        # Handle customer type with one-hot encoding
        df = pd.get_dummies(df, columns=["customer_type"], drop_first=False)
        dummy_cols = [col for col in df.columns if col.startswith("customer_type_")]
        df[dummy_cols] = df[dummy_cols].fillna(0).astype(int)
        
        # Drop unused columns
        df.drop(columns=["country", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type"], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Visualize the predicted ADR
def visualize_adr(df):
    try:
        st.write("### Visualizations of Predicted ADR")

        # Create a histogram of ADR distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting a histogram of ADR
        sns.histplot(df['adr'], kde=True, color='blue', ax=ax)

        # Set the title and labels
        ax.set_title("Distribution of ADR", fontsize=18)
        ax.set_xlabel("ADR", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)

        # Show the plot in Streamlit
        plt.tight_layout()
        st.pyplot(fig)
        
       


        # 1. Histogram of predicted ADR
    
        # Create a 'date' column from year, month, and day columns
        df['date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1))

        # Extract year and month
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Creating a figure for ADR over the years
        plt.figure(figsize=(12, 8))

        # Loop through each year and plot separately
        for year in [2015, 2016, 2017]:
            df_year = df[df['year'] == year].groupby('month')['adr'].mean().reset_index()
            # Shift the 'month' values by 1 position to the left
            df_year['month'] = df_year['month'] - 1
            df_year['month'] = df_year['month'].mod(12)  # Ensures wrapping of the months (e.g., Jan becomes Dec)
            sns.lineplot(data=df_year, x='month', y='adr', label=str(year), marker='o')

        # Set title and labels
        plt.title("Average ADR Over the Years", fontsize=18)
        plt.xlabel("Month", fontsize=14)  # Labeling as 'Month'
        plt.ylabel("Average ADR", fontsize=14)

        # Customizing x-axis ticks to show month names
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']

        # Set xticks and labels
        plt.xticks(ticks=range(12), labels=month_names, rotation=45)

        # Show the legend
        plt.legend(title="Year")

        # Show the plot
        st.pyplot(plt)




        # Set up the figure with two subplots (side by side)
        
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Line plot of average ADR by Arrival Month
        avg_adr_per_month = df.groupby('arrival_date_month')['adr'].mean()
        sns.lineplot(x=avg_adr_per_month.index, y=avg_adr_per_month.values, ax=axes[0], marker='o', color='blue')
        axes[0].set_title("Average ADR by Arrival Month", fontsize=18)
        axes[0].set_xlabel("Arrival Month", fontsize=14)
        axes[0].set_ylabel("Average ADR", fontsize=14)

        # Line plot of average ADR by Arrival Day of Month
        avg_adr_per_day = df.groupby('arrival_date_day_of_month')['adr'].mean()
        sns.lineplot(x=avg_adr_per_day.index, y=avg_adr_per_day.values, ax=axes[1], marker='o', color='blue')
        axes[1].set_title("Average ADR by Arrival Day of Month", fontsize=18)
        axes[1].set_xlabel("Arrival Day of Month", fontsize=14)
        axes[1].set_ylabel("Average ADR", fontsize=14)

        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)

       ## Create a new column for total guests
        df['total_guests'] = df['adults'] + df['children'] + df['babies']

        # Filter out total guests greater than 5 and exclude total_guests == 0
        df_filtered = df[(df['total_guests'] < 5) & (df['total_guests'] > 0)]

        # Get the value counts for total guests (up to 5 and not 0)
        total_guests_counts = df_filtered['total_guests'].value_counts().reset_index()

        # Rename the columns for better readability
        total_guests_counts.columns = ['Total Guests', 'Count']

        # Create a bar chart for total guests value counts
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use purple color for the bars
        sns.barplot(x='Total Guests', y='Count', data=total_guests_counts, ax=ax, color='purple')

        # Set the title and labels
        ax.set_title("Value Counts for Total Guests (1-5)", fontsize=18)
        ax.set_xlabel("Total Guests", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)

        # Show the plot in Streamlit
        plt.tight_layout()
        st.pyplot(fig)
        # Show the plot in Streamlit
        df['date'] = pd.to_datetime(df['date'])

        # Group by year and month, and calculate the total number of customers (adults + children + babies) per month
        df['year_month'] = df['date'].dt.to_period('M')

        # Sum the number of customers (adults + children + babies) per month
        df['total_customers'] = df['adults'] + df['children'] + df['babies']
        people_per_month = df.groupby('year_month')['total_customers'].sum()

        # Plotting the number of customers per month
        st.write("### Number of guests Per Month:")

        plt.figure(figsize=(12, 6))
        people_per_month.plot(kind='line', label='Total Customers', color='purple', linewidth=2)

        plt.title("Number of guests Per Month Over the Years")
        plt.xlabel("Month")
        plt.ylabel("Number of guests")
        plt.xticks(rotation=45)
        plt.legend()

        st.pyplot(plt)


        meal_mapping = {0: "Standard Meal", 1: "Breakfast", 2: "Half-Board", 3: "Full-Board"}

        # Reverse the mapping to get the original meal types
        df['meal'] = df['meal'].map(meal_mapping)

        # Ensure the 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Extract year from the date
        df['year'] = df['date'].dt.year

        # Group by year and meal type, and count the occurrences
        meal_count_per_year = df.groupby(['year', 'meal']).size().unstack(fill_value=0)

        # Plotting the meal types over the years
        st.write("### Meal Type Distribution Over the Years:")

        meal_count_per_year.plot(kind='bar', stacked=True, figsize=(12, 6))

        plt.title("Meal Type Distribution Over the Years")
        plt.xlabel("Year")
        plt.ylabel("Number of Meals")
        plt.xticks(rotation=45)
        plt.legend(title="Meal Type")

        st.pyplot(plt)
                
      
        # 2. Boxplot of predicted ADR
        st.write("#### Boxplot of Predicted ADR")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['adr'], color='green')
        plt.title("Boxplot of Predicted ADR", fontsize=18)
        plt.xlabel("Predicted ADR", fontsize=14)
        st.pyplot(plt)


    except Exception as e:
        st.error(f"Error during visualization: {e}")

# Main function for layout and interactivity
def main():
    # Custom styling for a beautiful look
    st.markdown("""
        <style>
            body {
                background: linear-gradient(to right, #f7b42c, #fc575e);
                font-family: 'Arial', sans-serif;
            }
            .main {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .header {
                font-size: 40px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 20px;
            }
            .sidebar {
                background-color: #ecf0f1;
            }
            .stButton {
                background-color: #2ecc71;
                border-radius: 5px;
                padding: 10px 20px;
                color: white;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .stButton:hover {
                background-color: #27ae60;
            }
            .stTextInput, .stFileUploader {
                background-color: #f1f1f1;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and Description
    st.title("Hotel ADR Prediction")
    st.markdown("""
        This app predicts the **Average Daily Rate (ADR)** for hotels based on your input dataset.
        Upload a dataset and let the model make predictions!
    """, unsafe_allow_html=True)

    # Sidebar for File Upload and Instructions
    with st.sidebar:
        st.header("Upload Your Data")
        st.markdown("""
        1. Upload your hotel dataset (CSV file).
        2. The model will predict the Average Daily Rate (ADR) based on various factors.
        3. You can download the processed dataset with predicted ADRs.
        """)
        
        uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
        model_path = "/Users/noachmeged/Documents/random_forest_model.pkl"
        df_mapping_path = "/Users/noachmeged/Downloads/country_adr_quantiles.csv"
    
    if uploaded_file is not None:
        try:
            # Load data and preprocess
            df = pd.read_csv(uploaded_file)
            df_mapping = pd.read_csv(df_mapping_path)
            
            # Create ADR dataset
            data = {"market_segment": ["Online TA", "Direct", "Aviation", "Offline TA/TO", "Groups", "Corporate", "Undefined", "Complementary"],
                    "adr": [117.186, 114.884, 100.142, 87.318, 79.432, 69.257, 15.000, 2.910]}
            df_adr = pd.DataFrame(data)
            df_adr["marketing_adr_bin"] = pd.qcut(df_adr["adr"], q=4, labels=[0, 1, 2, 3])
            
            adr_mean_per_channel = {"Corporate": 69.097646, "Direct": 106.559617, "GDS": 120.554301, "TA/TO": 103.469652}
            adr_df = pd.DataFrame(list(adr_mean_per_channel.items()), columns=["distribution_channel", "mean_adr"])
            adr_df["distribution_channel_adr_bin"] = pd.qcut(adr_df["mean_adr"], q=4, labels=[0, 1, 2, 3])
            
            # Preprocess the uploaded data
            df = preprocess_data(df, df_mapping, df_adr, adr_df)
            if df is None:
                return
            
            # Load and make predictions with the model
            random_forest_model = load_model(model_path)
            if random_forest_model is None:
                return
            
            # Reindex dataframe to match the model's input features
            model_features = random_forest_model.feature_names_in_
            df = df.reindex(columns=model_features, fill_value=0)
            
            # Make predictions
            df["adr"] = random_forest_model.predict(df).round(1)
            df.loc[(df["stays_in_weekend_nights"] == 0) & (df["stays_in_week_nights"] == 0), "adr"] = 0
            

            # Success notification combined with the descriptive statistics
            st.success("The predicted ADR has been processed successfully! Here's a summary:")

            # Display summary statistics for the predicted ADR
            adr_summary = df['adr'].describe()  # Summary statistics
            st.write(adr_summary)
           

            # Visualize the predicted ADR
            visualize_adr(df)
            
            # Provide option to download the processed CSV file
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predicted ADR CSV", csv, "predicted_adr.csv", "text/csv")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
# Run the app
if __name__ == "__main__":
    main()
