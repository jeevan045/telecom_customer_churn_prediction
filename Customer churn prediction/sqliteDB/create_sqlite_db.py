# create_sqlite_db.py
import sqlite3
import pandas as pd

# Load your CSV data
df = pd.read_csv('customer_churn_data.csv')

# Connect to SQLite database
conn = sqlite3.connect('database.db')

# Write the DataFrame to a SQL table
df.to_sql('customer_churn', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("SQLite database 'database.db' created successfully!")
