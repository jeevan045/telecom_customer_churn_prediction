import sqlite3
import pandas as pd
df = pd.read_csv('customer_churn_data.csv')

conn = sqlite3.connect('database.db')

df.to_sql('customer_churn', conn, if_exists='replace', index=False)

conn.close()

print("SQLite database 'database.db' created successfully!")

