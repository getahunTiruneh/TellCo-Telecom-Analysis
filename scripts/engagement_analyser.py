import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class EngagementAnalyzer:
    def __init__(self, df):
        self.df = df

    def user_engagement(self,df):
       # Calculate session frequency for each user
        session_frequency = df.groupby('MSISDN/Number').size().reset_index(name='Session Frequency')

        # Calculate duration of the session (already provided as 'Dur. (ms)')
        # If we need total session duration per user, you can sum it up
        session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Session Duration (ms)')

        # Calculate session total traffic
        session_traffic = df.groupby('MSISDN/Number').agg({
            'Total UL (Bytes)': 'sum',
            'Total DL (Bytes)': 'sum'
        }).reset_index()
        session_traffic.columns = ['MSISDN/Number', 'Total UL (Bytes)', 'Total DL (Bytes)']

        # Merge all metrics into a single DataFrame
        user_engagement = session_frequency.merge(session_duration, on='MSISDN/Number')
        user_engagement = user_engagement.merge(session_traffic, on='MSISDN/Number')

        # Display the final DataFrame with user engagement metrics
        return user_engagement
    def high_engagement_users(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]
        return high_engagement_users
        
    def plot_user_engagement(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]

        # Plot High Engagement Users

        # Set up the figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot High Engagement Users by Session Frequency
        sns.histplot(high_engagement_users['Session Frequency'], bins=50, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('High Engagement Users - Session Frequency')
        axs[0].set_xlabel('Session Frequency')
        axs[0].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Session Duration
        sns.histplot(high_engagement_users['Total Session Duration (ms)'], bins=50, kde=True, ax=axs[1], color='green')
        axs[1].set_title('High Engagement Users - Total Session Duration')
        axs[1].set_xlabel('Total Session Duration (ms)')
        axs[1].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Total Traffic
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        sns.histplot(high_engagement_users['Total Traffic (Bytes)'], bins=50, kde=True, ax=axs[2], color='red')
        axs[2].set_title('High Engagement Users - Total Traffic')
        axs[2].set_xlabel('Total Traffic (Bytes)')
        axs[2].set_ylabel('Number of High Engagement Users')

        # Adjust layout.
        plt.tight_layout()
        plt.show()
    def top_10_users_per_metric(self, df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        print("Top 10 Users by Session Frequency:\n", top_10_users_freq, "\n")
        print("Top 10 Users by Total Session Duration:\n", top_10_users_duration, "\n")
        print("Top 10 Users by Total Traffic:\n", top_10_users_traffic, "\n")
    
    
    def top_10_users(self,df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        return top_10_users_freq, top_10_users_duration, top_10_users_traffic
    
    
    def aggregate_traffic_per_user(self, df, applications):
        """
        Aggregates the traffic per user for the specified applications.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the traffic data.
        - applications (list of str): List of application columns to aggregate.

        Returns:
        - pd.DataFrame: A DataFrame with aggregated traffic per user.
        """
        return df.groupby('MSISDN/Number')[applications].sum().reset_index()

    def calculate_total_traffic(self, app_engagement, applications):
        """
        Calculates the total traffic for each application (DL + UL) and adds it to the DataFrame.

        Parameters:
        - app_engagement (pd.DataFrame): DataFrame with aggregated traffic per user.
        - applications (list of str): List of application columns to calculate total traffic for.

        Returns:
        - pd.DataFrame: Updated DataFrame with total traffic columns added.
        """
        for app in applications:
            total_col_name = app.replace(' (Bytes)', ' Total (Bytes)')
            app_engagement[total_col_name] = app_engagement[app]
        
        # Calculate total traffic for Social Media as an example
        app_engagement['Social Media Total (Bytes)'] = (
            app_engagement['Social Media DL Total (Bytes)'] + 
            app_engagement['Social Media UL Total (Bytes)']
        )
        
        return app_engagement

    def get_top_users(self, app_engagement, application, n=10):
        """
        Retrieves the top N users based on total traffic for a given application.

        Parameters:
        - app_engagement (pd.DataFrame): DataFrame with total traffic per user.
        - application (str): Application name for which to retrieve top users.
        - n (int): Number of top users to retrieve (default is 10).

        Returns:
        - pd.DataFrame: A DataFrame containing the top N users for the specified application.
        """
        column_name = f'{application} Total (Bytes)'
        return app_engagement.nlargest(n, column_name)
    
    def prepare_user_engagement_data(self,df):
        """
        Prepare user engagement data by aggregating and calculating necessary metrics.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing telecom data.
            
        Returns:
            pd.DataFrame: A DataFrame with aggregated user engagement metrics.
            pd.DataFrame: Top 10 users by session frequency.
            pd.DataFrame: Top 10 users by total duration.
            pd.DataFrame: Top 10 users by total traffic.
        """
        # Calculate total data volume per session
        df['Total Duration'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
        
        # Aggregate data by MSISDN/Number
        user_engagement_df = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # Number of sessions per user
            'Total Duration': 'sum',  # Total data volume of all sessions
            'Total UL (Bytes)': 'sum',  # Total upload bytes
            'Total DL (Bytes)': 'sum'  # Total download bytes
        }).reset_index()
        
        # Calculate the total traffic per user
        user_engagement_df['Total Traffic (Bytes)'] = user_engagement_df['Total UL (Bytes)'] + user_engagement_df['Total DL (Bytes)']
        
        # Rename columns for better understanding
        user_engagement_df.rename(columns={'Bearer Id': 'Session Frequency'}, inplace=True)
        
        # Find the top 10 customers per engagement metric
        top10_sessions = user_engagement_df.nlargest(10, 'Session Frequency')
        top10_duration = user_engagement_df.nlargest(10, 'Total Duration')
        top10_traffic = user_engagement_df.nlargest(10, 'Total Traffic (Bytes)')
        
        return user_engagement_df, top10_sessions, top10_duration, top10_traffic

    def apply_clustering(self, user_engagement_df, n_clusters=3):
        """
        Apply K-Means clustering to the user engagement data.
        
        Args:
            user_engagement_df (pd.DataFrame): The DataFrame containing user engagement metrics.
            n_clusters (int): The number of clusters for K-Means. Default is 3.
            
        Returns:
            pd.DataFrame: The DataFrame with an added 'Engagement Cluster' column.
        """
        # Selecting only the relevant columns for normalization
        metrics = ['Session Frequency', 'Total Duration', 'Total Traffic (Bytes)']
        
        # Normalize the selected metrics for clustering
        scaler = MinMaxScaler()
        user_engagement_df[metrics] = scaler.fit_transform(user_engagement_df[metrics])
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_engagement_df['Engagement Cluster'] = kmeans.fit_predict(user_engagement_df[metrics])
        
        return user_engagement_df
    def plot_top_applications(self, df, applications, top_n=3):
        """
        Summarize the total traffic for each application and plot the top N most used applications.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing traffic data for applications.
            applications (list): A list of column names representing different applications in the DataFrame.
            top_n (int): The number of top applications to plot. Default is 3.
        
        Returns:
            pd.Series: A Series containing the total traffic for each application.
        """
        # Sum up total traffic for each application
        app_usage = df[applications].sum().sort_values(ascending=False)
        
        # Plot the top N applications
        top_apps = app_usage.head(top_n)
        plt.figure(figsize=(12, 6))
        top_apps.plot(kind='bar', title=f'Top {top_n} Most Used Applications')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xlabel('Application')
        plt.xticks(rotation=45, ha='right')
        plt.show()
        
        return app_usage
    
    def plot_elbow_curve(self, df, metrics, max_k=10):
        """
        Determine the optimal number of clusters using the Elbow Method and plot the elbow curve.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to cluster.
            metrics (list): A list of column names to use for clustering.
            max_k (int): The maximum number of clusters to test. Default is 10.
        
        Returns:
            list: A list of distortions (inertia) for each number of clusters.
        """
        distortions = []
        
        # Iterate over the range of cluster numbers
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df[metrics])
            distortions.append(kmeans.inertia_)
        
        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Distortion (Inertia)')
        plt.grid(True)
        plt.show()
        
        return distortions