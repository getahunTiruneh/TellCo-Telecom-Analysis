import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

class PostgresConnection:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Retrieve database connection information from environment variables
        self.host = os.getenv('DB_HOST')
        self.database = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.port = os.getenv('DB_PORT')

        # Initialize connection and cursor
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print("Connected to PostgreSQL database")
        except (Exception, psycopg2.Error) as error:
            print("Error connecting to PostgreSQL database:", error)

    def disconnect(self):
        """Closes the connection to the PostgreSQL database."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Connection closed!")

    def execute_query(self, query):
        """Executes a single query and commits changes."""
        try:
            self.cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except (Exception, psycopg2.Error) as error:
            print("Error executing query:", error)

    def fetch_data(self, query):
        """Executes a SELECT query and returns the result as a pandas DataFrame."""
        try:
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=column_names)
            return df
        except (Exception, psycopg2.Error) as error:
            print("Error fetching data:", error)
            return None