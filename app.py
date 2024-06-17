import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Supply Chain Optimization")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    MPSCData = pd.read_csv(uploaded_file)
    
    st.write("Data Preview")
    st.dataframe(MPSCData.head())

    # Data Cleaning
    MPSCData.columns = [col.lower().replace(' ', '_') for col in MPSCData.columns]
    MPSCData.rename(columns=lambda x: x.replace("(", "").replace(")", ""), inplace=True)
    
    # Removing non-numeric SKU values
    MPSCData['sku'] = MPSCData['sku'].str.extract('(\d+)').astype(float)

    # Encoding categorical features
    categorical_cols = ['product_type', 'customer_demographics', 'location', 'transportation_modes', 'routes']
    le = LabelEncoder()
    for col in categorical_cols:
        MPSCData[col] = le.fit_transform(MPSCData[col])
    
    # Handling non-numeric inspection results
    MPSCData['inspection_results'] = le.fit_transform(MPSCData['inspection_results'])

    # Feature Selection
    features = ['sku', 'price', 'revenue_generated', 'lead_times', 'shipping_times', 'shipping_costs', 'lead_time', 
                'production_volumes', 'manufacturing_lead_time', 'manufacturing_costs', 'inspection_results', 
                'defect_rates', 'routes', 'costs']
    X = MPSCData[features]
    Y = MPSCData['stock_levels']

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training the model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)

    # Predictions
    Y_pred = model.predict(X_test_scaled)

    # Model Evaluation
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    
    st.write(f"Model R-squared: {r2}")
    st.write(f"Model Mean Squared Error: {mse}")

    # Plotting the results
    st.subheader("Actual vs Predicted Stock Levels")
    fig, ax = plt.subplots()
    sns.scatterplot(x=Y_test, y=Y_pred, ax=ax)
    sns.regplot(x=Y_test, y=Y_pred, scatter=False, ax=ax)
    plt.xlabel('Actual Stock Levels')
    plt.ylabel('Predicted Stock Levels')
    plt.title('Actual vs Predicted Stock Levels')
    st.pyplot(fig)

# Data Analysis Page
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ("Home", "Data Analysis", "Stock Level Prediction"))

if page == "Data Analysis":
    st.title("Data Analysis")

    # Distribution of Stock Levels
    st.write("Distribution of Stock Levels")
    fig, ax = plt.subplots()
    sns.histplot(MPSCData['stock_levels'], kde=True, ax=ax)
    plt.title('Distribution of Stock Levels')
    st.pyplot(fig)

    st.write("Product Analysis")
    fig, axes = plt.subplots(2, 2, figsize=(12, 14))
    MPSCData.groupby('product_type')['stock_levels'].sum().sort_values(ascending=False).plot.bar(ax=axes[0, 0], title="Total Stock")
    MPSCData.groupby('product_type')['order_quantities'].sum().sort_values(ascending=False).plot.bar(ax=axes[0, 1], title="Total Order")
    MPSCData.groupby('product_type')['manufacturing_costs'].sum().sort_values(ascending=False).plot.bar(ax=axes[1, 0], title="Manufacturing Costs")
    MPSCData.groupby('product_type')['revenue_generated'].sum().sort_values(ascending=False).plot.bar(ax=axes[1, 1], title="Revenue")
    plt.tight_layout()
    st.pyplot(fig)

    # Scatterplot of Price vs Stock Levels
    st.write("Price vs Stock Levels")
    fig, ax = plt.subplots()
    sns.scatterplot(x='price', y='stock_levels', data=MPSCData, ax=ax)
    plt.title('Price vs Stock Levels')
    st.pyplot(fig)

    st.write("Stock Levels by Product Type")
    fig, ax = plt.subplots()
    sns.boxplot(x='product_type', y='stock_levels', data=MPSCData, ax=ax)
    plt.title('Stock Levels by Product Type')
    st.pyplot(fig)

if page == "Stock Level Prediction":
    st.title("Predict Stock Levels")
    sku = st.number_input("SKU", min_value=0, step=1)
    price = st.number_input("Price")
    revenue_generated = st.number_input("Revenue Generated")
    lead_times = st.number_input("Lead Times")
    shipping_times = st.number_input("Shipping Times")
    shipping_costs = st.number_input("Shipping Costs")
    lead_time = st.number_input("Lead Time")
    production_volumes = st.number_input("Production Volumes")
    manufacturing_lead_time = st.number_input("Manufacturing Lead Time")
    manufacturing_costs = st.number_input("Manufacturing Costs")
    inspection_results = st.selectbox("Inspection Results", le.classes_)
    defect_rates = st.number_input("Defect Rates")
    routes = st.number_input("Routes", min_value=0, step=1)
    costs = st.number_input("Costs")

    if st.button("Predict Stock Levels"):
        inspection_results_encoded = le.transform([inspection_results])[0]
        new_data = np.array([sku, price, revenue_generated, lead_times, shipping_times, shipping_costs, lead_time,
                             production_volumes, manufacturing_lead_time, manufacturing_costs, inspection_results_encoded,
                             defect_rates, routes, costs]).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        st.write(f"Predicted Stock Level: {prediction[0]}")