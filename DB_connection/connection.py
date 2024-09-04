import psycopg2
import pandas as pd

class PostgresConnection:
    def __init__(self,  database, user, password,host='localhost', port='5432'):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.cursor = None

    def connect(self):
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
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Connection closed!")

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except (Exception, psycopg2.Error) as error:
            print("Error executing query:", error)

    def fetch_data(self, query):
        try:
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=column_names)
            return df
        except (Exception, psycopg2.Error) as error:
            print("Error fetching data:", error)
            return None