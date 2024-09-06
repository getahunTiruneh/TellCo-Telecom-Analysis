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
    
    
    