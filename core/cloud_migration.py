import pandas as pd
from sqlalchemy import create_engine
import toml

print("🚀 Starting Cloud Migration...")

# 1. Load your secret TiDB credentials
try:
    secrets = toml.load(".streamlit/secrets.toml")
    db_url = secrets["connections"]["cloud_db"]["url"]
except Exception as e:
    print(f"❌ Error loading secrets: {e}")
    print("Make sure .streamlit/secrets.toml exists and has the correct format.")
    exit(1)

# 2. Connect to the Cloud Database
print("🔌 Connecting to TiDB Cloud...")
engine = create_engine(db_url)

# 3. Load the new 106-skill dataset
try:
    df = pd.read_csv("data/engineered_features.csv")
    print(f"📊 Loaded {len(df)} skills from local CSV.")
except FileNotFoundError:
    print("❌ Error: data/engineered_features.csv not found. Did you run ml_engine.py?")
    exit(1)

# 4. Push to TiDB (Replacing the old table to fix the 0 bug and add Domains)
print("☁️ Uploading data to TiDB database. This might take a few seconds...")
df.to_sql("skill_features", con=engine, if_exists="replace", index=False)

print("✅ Migration complete! Database now has 106 skills and precise decimal forecasts.")