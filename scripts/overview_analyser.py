import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class UserOverviewAnalysis:
    def __init__(self,df):
        self.df=df
    def top_10_handsets(self,df):
        # Count occurrences of each handset (Handset Type)
        handset_counts = df['Handset Type'].value_counts()
        
       #Get top 10 handsets counts
        top_10_handsets=handset_counts.head(10)
        print("Top 10 Handsets Used by Customers:")
        return top_10_handsets.to_frame().reset_index()
    def top_3_handset_manufacturers(self, df):
        """
        Plots the top 3 handset manufacturers based on the number of users.

        Args:
        df (pd.DataFrame): The input DataFrame containing handset information with a column 'Handset Manufacturer'.

        Returns:
        pd.DataFrame: A DataFrame with the top 3 handset manufacturers and their user counts.
        """
        # Count occurrences of each handset manufacturer
        handset_manufacturer_counts = df['Handset Manufacturer'].value_counts()

        # Get top 3 handset manufacturers
        top_3_handset_manufacturers = handset_manufacturer_counts.head(3)
        top_3_handset_manufacturers.to_frame().reset_index()
        # Print the top 3 handset manufacturers
        print("Top 3 Handset Manufacturers:")
        return top_3_handset_manufacturers
    def plot_top_3_handset_manufacturers(self,df):
        handset_manufacturer_counts=df['Handset Manufacturer'].value_counts()
        # Get top 3 handset manufacturers
        top_3_handset_manufacturers = handset_manufacturer_counts.head(3)
        
        # Plot the top 3 handset manufacturers
        top_3_handset_manufacturers.plot(kind='pie', legend=False, color='skyblue')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Handset Manufacturers')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def top_5_handsets_per_manufacturer(self, df):
        # Call top_3_handset_manufacturers to get 'top_3_manufacturers' without displaying it
        top_3_manufacturers = self.top_3_handset_manufacturers(df)
        
        # Filter dataset for top 3 manufacturers
        top_3_manufacturers_list = top_3_manufacturers.iloc[:,0].tolist()
        filtered_df = df[df['Handset Manufacturer'].isin(top_3_manufacturers_list)]

        # Identify top 5 handsets per manufacturer
        top_5_handsets_per_manufacturer = {}
        for manufacturer in top_3_manufacturers_list:
            manufacturer_data = filtered_df[filtered_df['Handset Manufacturer'] == manufacturer]
            top_5_handsets = manufacturer_data['Handset Type'].value_counts().head(5)
            top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

        # Print the top 5 handsets per manufacturer
        print("\nTop 5 Handsets per Top 3 Manufacturer:")
        for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
            print(f"\n{manufacturer}:")
            print(handsets)
        return top_5_handsets_per_manufacturer
    
    def plot_top_5_handsets_per_manufacturer(self, df):
        top_5_handsets_per_manufacturer=self.top_5_handsets_per_manufacturer(df)
        # Plot the top 5 handsets per manufacturer
        for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
            plt.figure(figsize=(10, 6))
            handsets.plot(kind='bar', edgecolor='black')
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handset Type')
            plt.ylabel('Number of Users')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        
