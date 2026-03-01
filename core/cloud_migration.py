import pandas as pd
from sqlalchemy import create_engine
import toml
import os

def migrate_to_mysql():
    print("☁️ Booting up Cloud MySQL Migration Module...")
    
    # 1. Load your secure local credentials
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        print("🚨 Error: .streamlit/secrets.toml not found! Please create it.")
        return
        
    secrets = toml.load(secrets_path)
    db_url = secrets['connections']['cloud_db']['url']
    
    # Ensure it uses the pymysql driver
    if db_url.startswith("mysql://"):
        db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)
        
    # 2. Connect to the Cloud Server
    print("🔄 Establishing secure connection to the MySQL cloud...")
    engine = create_engine(db_url)
    
    # 3. Load the local engineered data
    df = pd.read_csv('data/engineered_features.csv')
    
    # 4. Push the data to the cloud
    print(f"📤 Uploading {len(df)} records to the cloud. This creates your SQL table automatically...")
    
    try:
        df.to_sql('skill_features', con=engine, if_exists='replace', index=False)
        print("✅ Migration Complete! Your system's data is now hosted securely in the cloud.")
    except Exception as e:
        print(f"🚨 Migration Failed: {e}")

if __name__ == "__main__":
    migrate_to_mysql()