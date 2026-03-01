import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_dataset():
    # 1. Load the data from our new DB
    conn = sqlite3.connect('data/skill_decay.db')
    df = pd.read_sql_query("SELECT * FROM master_trends", conn)
    
    # 2. Separate the 'skill' names from the numeric data
    skills = df[['skill']]
    numeric_data = df.drop(columns=['skill'])
    
    # 3. Apply Min-Max Scaling (0 to 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(numeric_data)
    
    # 4. Rebuild the DataFrame
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_data.columns)
    final_df = pd.concat([skills, scaled_df], axis=1)
    
    # 5. Save the 'Golden Dataset' for the ML model
    if not os.path.exists('data'): os.makedirs('data')
    final_df.to_csv('data/normalized_skills.csv', index=False)
    
    print("\n✅ Data Normalization Complete!")
    print("Your 'Golden Dataset' is ready at data/normalized_skills.csv")
    print(final_df.head()) # Show the first 5 rows
    
    conn.close()

if __name__ == "__main__":
    normalize_dataset()