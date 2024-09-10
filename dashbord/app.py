import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Set page configuration
st.set_page_config(page_title="Telecom Data - Exploratory Data Analysis", page_icon=":bar_chart:", layout="wide")

# Database connection configuration
db_config = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'database': 'teleco'
}

# Create SQLAlchemy engine and load data from PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
df = pd.read_sql("SELECT * FROM xdr_data", engine)

# Set page title
st.title("Telecom Data - Exploratory Data Analysis")

# Show first few rows of the dataset
st.subheader("Initial Data")
st.write(df.head())

# Sidebar Menu
st.sidebar.title("📊 Telecom Data Analysis App")
st.sidebar.markdown("Use the menu below to navigate through different analysis sections:")

menu_options = {
    "📈 User Overview Analysis": "Overview of user behavior and handset usage patterns.",
    "📊 User Engagement Analysis": "Engagement metrics for network services usage.",
    "📉 User Experience Analysis": "Analysis of network performance and user experience.",
    "😊 User Satisfaction Analysis": "Assessment of user satisfaction based on network metrics."
}

# Sidebar radio buttons with descriptions
selected_option = st.sidebar.radio("Select Analysis Type", list(menu_options.keys()))

# User Overview Analysis Section
if selected_option == "📈 User Overview Analysis":
    st.subheader("📈 User Overview Analysis")
    st.write(menu_options[selected_option])
    
    # Top 10 Handsets used by customers
    top_handsets = df['Handset Type'].value_counts().nlargest(10)
    st.write("Top 10 Handsets Used by Customers")
    st.bar_chart(top_handsets)

    # Top 3 Handset Manufacturers
    top_manufacturers = df['Handset Manufacturer'].value_counts().nlargest(3)
    st.write("Top 3 Handset Manufacturers")
    st.bar_chart(top_manufacturers)

    # Top 5 Handsets per Top 3 Handset Manufacturers
    st.write("Top 5 Handsets per Top 3 Handset Manufacturers")
    for manufacturer in top_manufacturers.index:
        st.write(f"Handsets for {manufacturer}")
        top_handsets_per_manufacturer = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().nlargest(5)
        st.bar_chart(top_handsets_per_manufacturer)

# User Engagement Analysis Section
elif selected_option == "📊 User Engagement Analysis":
    st.subheader("📊 User Engagement Analysis")
    st.write(menu_options[selected_option])

    # Visualize network parameters such as 'Avg RTT DL (ms)', 'Avg RTT UL (ms)'
    st.write("Average Round Trip Time (RTT) Downlink and Uplink")
    fig, ax = plt.subplots()
    sns.histplot(df['Avg RTT DL (ms)'], ax=ax, kde=True, color='blue', label='RTT DL')
    sns.histplot(df['Avg RTT UL (ms)'], ax=ax, kde=True, color='orange', label='RTT UL')
    plt.legend()
    st.pyplot(fig)

# User Experience Analysis Section
elif selected_option == "📉 User Experience Analysis":
    st.subheader("📉 User Experience Analysis")
    st.write(menu_options[selected_option])

    # Analyze 'Avg Bearer TP DL (kbps)' and 'Avg Bearer TP UL (kbps)'
    st.write("Average Bearer Throughput Downlink and Uplink")
    fig, ax = plt.subplots()
    sns.histplot(df['Avg Bearer TP DL (kbps)'], ax=ax, kde=True, color='green', label='Bearer TP DL')
    sns.histplot(df['Avg Bearer TP UL (kbps)'], ax=ax, kde=True, color='red', label='Bearer TP UL')
    plt.legend()
    st.pyplot(fig)

# User Satisfaction Analysis Section
elif selected_option == "😊 User Satisfaction Analysis":
    st.subheader("😊 User Satisfaction Analysis")
    st.write(menu_options[selected_option])
    
    # Analyze 'TCP DL Retrans. Vol (Bytes)' and 'TCP UL Retrans. Vol (Bytes)'
    st.write("TCP Downlink and Uplink Retransmission Volumes")
    fig, ax = plt.subplots()
    sns.histplot(df['TCP DL Retrans. Vol (Bytes)'], ax=ax, kde=True, color='purple', label='TCP DL Retrans. Vol')
    sns.histplot(df['TCP UL Retrans. Vol (Bytes)'], ax=ax, kde=True, color='brown', label='TCP UL Retrans. Vol')
    plt.legend()
    st.pyplot(fig)
