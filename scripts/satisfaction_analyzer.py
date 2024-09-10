import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

class SatisfactionAnalyer:
    def __init__(self,df):
        self.df=df
        
    def user_engagement(self,df):
        df['Total Duration']=df['Total UL (Bytes)']+df['Total DL (Bytes)']
        # Assume df is the DataFrame containing the dataset
        engagement_df = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # This will give us the number of sessions per user
            'Total Duration': 'sum',  # Total duration of all sessions
            'Total UL (Bytes)': 'sum',  # Total upload bytes
            'Total DL (Bytes)': 'sum',  # Total download bytes
        }).reset_index()

        # Calculate the total traffic per user
        engagement_df['Total Traffic (Bytes)'] = engagement_df['Total UL (Bytes)'] + engagement_df['Total DL (Bytes)']

        # Rename columns for better understanding
        engagement_df.rename(columns={'Bearer Id': 'Session Frequency'}, inplace=True)
        
        # Selecting only the relevant columns for normalization
        metrics = ['Session Frequency', 'Total Duration', 'Total Traffic (Bytes)']
        scaler = MinMaxScaler()
        engagement_df[metrics] = scaler.fit_transform(engagement_df[metrics])

        # Applying K-Means clustering with k=3
        kmeans = KMeans(n_clusters=3, random_state=42)
        engagement_df['Engagement Cluster'] = kmeans.fit_predict(engagement_df[metrics])
        return engagement_df
    def get_least_engaged_cluster(self, df, cluster_column, metrics):
        """
        Determine the cluster with the least engagement based on specified metrics.

        Parameters:
        df (DataFrame): The input DataFrame containing the clustered data.
        cluster_column (str): The column name representing the cluster labels.
        metrics (list): List of metrics to consider for determining least engagement.

        Returns:
        int: The cluster number with the least engagement.
        """
        # Calculate the centroids of each engagement cluster
        engagement_centroids = df.groupby(cluster_column)[metrics].mean()

        # Display the centroids for each cluster
        print(engagement_centroids)

        # Sum the normalized metrics for each cluster to get a measure of total engagement
        engagement_centroids['Total Engagement Score'] = engagement_centroids.sum(axis=1)

        # Identify the cluster with the lowest total engagement score
        least_engaged_cluster = engagement_centroids['Total Engagement Score'].idxmin()

        print(f"The least engaged cluster is: {least_engaged_cluster}")

        return least_engaged_cluster
    def get_worst_experience_cluster(self, df, cluster_column, metrics):
        """
        Determine the cluster with the worst experience based on specified metrics.
        
        Parameters:
        df (DataFrame): The input DataFrame containing the clustered data.
        cluster_column (str): The column name representing the cluster labels.
        metrics (list): List of metrics to consider for determining worst experience.
        
        Returns:
        int: The cluster number with the worst experience.
        """
        # Compute the mean values for each cluster
        cluster_means = df.groupby(cluster_column)[metrics].mean()

        # Determine the worst experience cluster
        # Define worst experience based on highest average values for the metrics
        # Example: Highest average RTT, TCP retransmission, etc.
        cluster_means['Average'] = cluster_means.mean(axis=1).idxmax()
        # Determine the worst experience cluster based on the highest average
        worst_experience_cluster = cluster_means['Average'].idxmax()

        print("Worst Experience Cluster:", worst_experience_cluster)
        return worst_experience_cluster

    # Function to calculate Euclidean distance
    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    # Task 4.1: Assign engagement and experience scores
    def calculate_scores(self, engagement_df, experience_df):
        
        # Remove 'MSISDN/Number' and 'Customer Number' it is not used in calculations
        engagement_columns = [col for col in engagement_df.columns if col != 'MSISDN/Number']
        experience_columns = [col for col in experience_df.columns if col != 'Customer Number']
        
        # Calculate the center of each engagement cluster
        engagement_centers = engagement_df.groupby('Engagement Cluster')[engagement_columns].mean()
        
        # Calculate engagement scores
        engagement_scores = []
        for _, row in engagement_df.iterrows():
            cluster_center = engagement_centers.loc[row['Engagement Cluster']]
            score = self.euclidean_distance(row[engagement_columns], cluster_center)
            engagement_scores.append(score)
        
        engagement_df['Engagement Score'] = engagement_scores
        
        # Calculate the center of each experience cluster
        numeric_cols = experience_df.select_dtypes(include=[np.number]).columns
        experience_centers = experience_df[numeric_cols].groupby('Experience Cluster').mean().drop(columns=['MSISDN/Number'], errors='ignore')
        
        # Calculate experience scores
        experience_scores = []
        for _, row in experience_df.iterrows():
            cluster_center = experience_centers.loc[row['Experience Cluster']]
            score = self.euclidean_distance(row[experience_columns], cluster_center)
            experience_scores.append(score)
        
        experience_df['Experience Score'] = experience_scores
        
        return engagement_df, experience_df

    # Task 4.2: Calculate satisfaction score
    def calculate_satisfaction(self, engagement_df, experience_df):
        
        experience_df.rename(columns={'Customer Number': 'MSISDN/Number'}, inplace=True)
        # Merge engagement and experience data
        merged_df = engagement_df.merge(experience_df, on='MSISDN/Number', suffixes=('_engagement', '_experience'))
        
        # Calculate satisfaction score
        merged_df['Satisfaction Score'] = (merged_df['Engagement Score'] + merged_df['Experience Score']) / 2
        
        # Get top 10 satisfied customers
        top_10_satisfied = merged_df.nlargest(10, 'Satisfaction Score')
        return merged_df, top_10_satisfied
    
    def plot_top_10_satisfied(self, top_10_satisfied):
        plt.figure(figsize=(10, 6))
        sns.barplot( x='MSISDN/Number',y='Satisfaction Score', data=top_10_satisfied)
        plt.title('Top 10 Most Satisfied Customers')
        plt.ylabel('Satisfaction Score')
        plt.xlabel('Customer ID')
        plt.xticks(rotation=45)
        plt.show()

    # Task 4.3: Build a regression model
    def build_regression_model(self, merged_df):
        """
        Build a linear regression model to predict the satisfaction score.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing 'Engagement Score', 'Experience Score', and 'Satisfaction Score'.

        Returns:
        - model (LinearRegression): Trained Linear Regression model.
        - coefficients (array): Coefficients of the regression model.
        - intercept (float): Intercept of the regression model.
        """
        # Define features (X) and target (y)
        X = merged_df[['Engagement Score', 'Experience Score']]
        y = merged_df['Satisfaction Score']
        
        # Initialize the Linear Regression model
        model = LinearRegression()
        
        # Fit the model to the data
        model.fit(X, y)
        
        # Print regression model coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        
        print("Regression model coefficients:", coefficients)
        print("Regression model intercept:", intercept)
        
        return model, coefficients, intercept
    def perform_kmeans_clustering(self, merged_df, n_clusters=2, random_state=42):
        """
        Perform K-means clustering on the engagement and experience scores.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing 'Engagement Score' and 'Experience Score'.
        - n_clusters (int): Number of clusters for K-means. Default is 2.
        - random_state (int): Random state for reproducibility. Default is 42.

        Returns:
        - merged_df (pd.DataFrame): DataFrame with an additional 'Cluster' column indicating cluster assignments.
        - kmeans (KMeans): Trained KMeans model.
        """
        # Extract features for clustering
        X_cluster = merged_df[['Engagement Score', 'Experience Score']]
        
        # Standardize the features
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        merged_df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
        
        return merged_df, kmeans
    def aggregate_scores_per_cluster(self, merged_df):
        """
        Aggregate the average satisfaction and experience scores per cluster.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing 'Cluster', 'Satisfaction Score', and 'Experience Score'.

        Returns:
        - cluster_aggregation (pd.DataFrame): DataFrame with mean satisfaction and experience scores per cluster.
        """
        # Aggregate the average satisfaction and experience score per cluster
        cluster_aggregation = merged_df.groupby('Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        })

        return cluster_aggregation
    
    def export_to_postgresql(self, merged_df, table_name, db_config):
        """
        Exports the final table containing user ID, engagement, experience, and satisfaction scores to MySQL database.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing the final table data.
        - table_name (str): Name of the table in the MySQL database.
        - db_config (dict): Dictionary containing database configuration details like user, password, host, database.

        Returns:
        - None
        """
        # Create a connection to the MySQL database
        try:
            # Define the SQLAlchemy engine for PostgreSQL
            engine = create_engine(
                f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
            )
            # Export the DataFrame to the PostgreSQL table
            merged_df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
            print(f"Data exported successfully to the table '{table_name}' in the PostgreSQL database.")

        except Exception as e:
            print("Error occurred while exporting data to PostgreSQL:", e)