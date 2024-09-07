import pandas as pd
import numpy as np 

class DataQualityCheck:
    def __init__(self,df):
        self.df=df
    def check_missing_data(self,df):
        # checking for missing value in each columns
        missing_data_summary = (df.isnull().sum() / len(df) * 100).round(2).sort_values(ascending=False)
        missing_data_summary = missing_data_summary.to_frame().reset_index()
        missing_data_summary.columns = ["Feature", "Missing in %"]
        
        missing_data_summary = missing_data_summary[missing_data_summary['Missing in %'] > 0]
        if missing_data_summary.empty:
            return "Success: No missing data!"
        else:
            return missing_data_summary
    def check_duplicate_data(self, df):
        duplicate_rows = df[df.duplicated()]
        if duplicate_rows.empty:
            return "Success: No duplicate data!"
        else:
            return duplicate_rows
    def fix_outliers(self, df, columns):
        # remove outliers for the selected columns
        for column in columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        return df
    
    
    