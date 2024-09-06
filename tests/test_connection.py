import pytest
import os
import sys
# Get the current working directory
current_dir = os.getcwd()

# Append the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DB_connection.connection import PostgresConnection

# Fixture for setting up and tearing down the database connection
@pytest.fixture(scope='module')
def db_connection():
    # Initialize the connection instance
    db = PostgresConnection()
    
    # Establish the connection
    db.connect()
    
    # Provide the connection to the tests
    yield db
    
    # Close the connection after tests are done
    db.disconnect()

# Test to check if the connection is established successfully
def test_connect(db_connection):
    assert db_connection.connection is not None, "Connection should be established"
    assert db_connection.cursor is not None, "Cursor should be initialized"

