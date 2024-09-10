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
    # Calculate the mean values for each cluster
    cluster_means = df.groupby(cluster_column)[metrics].mean()
    
    # Define criteria for worst experience (example: highest average values for negative metrics)
    # For instance, you might want to maximize values for RTT and TCP retransmission
    worst_experience = cluster_means[metrics].max().idxmax()  # Cluster with the highest average value in the metrics
    
    return worst_experience

# Function to calculate Euclidean distance
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Task 4.1: Assign engagement and experience scores
def calculate_scores(engagement_df, experience_df, engagement_clusters, experience_clusters):
    # Calculate the center of each engagement cluster
    engagement_centers = engagement_df.groupby('Engagement Cluster').mean().drop(columns=['UserID'])
    
    # Calculate engagement scores
    engagement_scores = []
    for index, row in engagement_df.iterrows():
        cluster_center = engagement_centers.loc[row['Engagement Cluster']]
        score = euclidean_distance(row[engagement_df.columns != 'UserID'], cluster_center)
        engagement_scores.append(score)
    
    engagement_df['Engagement Score'] = engagement_scores
    
    # Calculate the center of each experience cluster
    experience_centers = experience_df.groupby('Experience Cluster').mean().drop(columns=['UserID'])
    
    # Calculate experience scores
    experience_scores = []
    for index, row in experience_df.iterrows():
        cluster_center = experience_centers.loc[row['Experience Cluster']]
        score = euclidean_distance(row[experience_df.columns != 'UserID'], cluster_center)
        experience_scores.append(score)
    
    experience_df['Experience Score'] = experience_scores
    
    return engagement_df, experience_df

# Task 4.2: Calculate satisfaction score
def calculate_satisfaction(engagement_df, experience_df):
    # Merge engagement and experience data
    merged_df = engagement_df.merge(experience_df, on='UserID', suffixes=('_engagement', '_experience'))
    
    # Calculate satisfaction score
    merged_df['Satisfaction Score'] = (merged_df['Engagement Score'] + merged_df['Experience Score']) / 2
    
    # Get top 10 satisfied customers
    top_10_satisfied = merged_df.nlargest(10, 'Satisfaction Score')
    return top_10_satisfied

# Task 4.3: Build a regression model
def build_regression_model(df):
    X = df[['Engagement Score', 'Experience Score']]
    y = df['Satisfaction Score']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Task 4.4: K-means clustering on engagement & experience scores
def kmeans_clustering(df):
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Score Cluster'] = kmeans.fit_predict(df[['Engagement Score', 'Experience Score']])
    return df
