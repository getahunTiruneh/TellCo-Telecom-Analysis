import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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