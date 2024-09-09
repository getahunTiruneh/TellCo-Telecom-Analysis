import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class ExperienceAnalyzer:
    def __init__(self, df):
        self.df = df

    def fill_missing_values(self, df):
        # Replace missing values with mean for numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        # Replace missing values with mode for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        return df
    def aggregate_customer_data(self, df):
        """
        Aggregates the required information per customer (MSISDN/Number).
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        
        Returns:
        DataFrame: Aggregated DataFrame with mean values for numerical columns and the first entry for categorical columns.
        """
        aggregated_df = df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Taking the first handset type per customer
        }).reset_index()

        return aggregated_df

    def get_top_bottom_most_freq_values(self, df, column_name, top_n=10):
        """
        Get the top N, bottom N, and most frequent N values for a specified column.

        Args:
        df (DataFrame): The input DataFrame containing the data.
        column_name (str): The name of the column to analyze.
        top_n (int): The number of top, bottom, and most frequent values to retrieve. Default is 10.

        Returns:
        DataFrame: A DataFrame with the top N, bottom N, and most frequent N values for the specified column.
        """
        # Get the top, bottom, and most frequent values
        top_values = df[column_name].nlargest(top_n).reset_index(name=f'Top {column_name}')
        bottom_values = df[column_name].nsmallest(top_n).reset_index(name=f'Bottom {column_name}')
        most_freq_values = df[column_name].value_counts().nlargest(top_n).reset_index(name=f'Most Frequent {column_name}')
        
        # Rename columns for clarity
        top_values.columns = ['Index', f'Top {column_name}']
        bottom_values.columns = ['Index', f'Bottom {column_name}']
        most_freq_values.columns = [f'Most Frequent {column_name}', 'Frequency']

        # Merge results into a single DataFrame
        result_df = pd.concat([top_values, bottom_values, most_freq_values], axis=1)
        
        return result_df

    def plot_top_10_throughput_distribution(self, df, column_name='Avg Bearer TP DL (kbps)', top_n=10):
        """
        Plot the distribution of the top 10 average throughput per handset type.
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        column_name (str): The column name for average throughput. Default is 'Avg Bearer TP DL (kbps)'.
        top_n (int): The number of top handset types to consider. Default is 10.
        """
        top_throughput_df = df.sort_values(by=column_name, ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Handset Type', y=column_name, data=top_throughput_df)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of {column_name} per Handset Type')
        plt.show()

    def plot_top_10_tcp_retransmission(self, df, column_name='TCP DL Retrans. Vol (Bytes)', top_n=10):
        """
        Plot the top N average TCP retransmission per handset type.
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        column_name (str): The column name for TCP retransmission. Default is 'TCP DL Retrans. Vol (Bytes)'.
        top_n (int): The number of top handset types to consider. Default is 10.
        """
        avg_tcp_retrans_per_handset = df.groupby('Handset Type')[column_name].mean().reset_index()
        top_tcp_retrans_df = avg_tcp_retrans_per_handset.sort_values(by=column_name, ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y=column_name, data=top_tcp_retrans_df)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Average {column_name} per Handset Type')
        plt.show()
    
    def perform_clustering(self, df, features_columns, n_clusters=3):
        """
        Perform clustering on the DataFrame using K-Means and describe each cluster.

        Args:
        df (DataFrame): The input DataFrame containing the data.
        features_columns (list): List of column names to be used for clustering.
        n_clusters (int): The number of clusters for K-Means. Default is 3.

        Returns:
        DataFrame: A DataFrame describing each cluster with mean values for numeric columns.
        """
        # Selecting relevant columns for clustering
        features = df[features_columns]

        # Normalizing the data
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        # Applying K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Experience Cluster'] = kmeans.fit_predict(features_scaled)

        # Select only numeric columns for the cluster description
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Describing each cluster
        cluster_description = df.groupby('Experience Cluster')[numeric_columns].mean()

        return cluster_description
    
    def plot_2d_clusters(self, df, feature1, feature2, cluster_column='Experience Cluster'):
        """
        Plot a 2D scatter plot of clusters.

        Args:
        df (DataFrame): The input DataFrame containing the data with cluster assignments.
        feature1 (str): The name of the first feature to plot.
        feature2 (str): The name of the second feature to plot.
        cluster_column (str): The column name for cluster assignments. Default is 'Experience Cluster'.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature1], y=df[feature2], hue=df[cluster_column], palette='viridis', s=100, alpha=0.7)
        plt.title('2D Scatter Plot of Clusters')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.legend(title='Cluster')
        plt.show()