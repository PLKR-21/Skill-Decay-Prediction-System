import requests
import pandas as pd
import sqlite3
import time

class SkillDataFactory:
    def __init__(self):
        # 21 Diverse Skills across Web, Mobile, AI, and Languages
        self.skills = [
            'python', 'javascript', 'java', 'rust', 'go', 'typescript',
            'reactjs', 'angular', 'vue.js', 'jquery', 'next.js',
            'tensorflow', 'pytorch', 'flutter', 'kotlin', 'swift',
            'php', 'ruby', 'c++', 'pandas', 'svelte'
        ]
        # Unix timestamps for 2023, 2024, 2025
        self.years = {
            '2023': (1672531200, 1704067199), 
            '2024': (1704067200, 1735689599), 
            '2025': (1735689600, 1767225599)
        }
        self.db_path = 'data/skill_decay.db'

    def fetch_so(self, skill, start, end):
        url = f"https://api.stackexchange.com/2.3/questions?fromdate={start}&todate={end}&tagged={skill}&site=stackoverflow&filter=total"
        try:
            res = requests.get(url).json()
            return res.get('total', 0)
        except: return 0

    def fetch_gh(self, skill, year):
        query = f"{skill}+created:{year}-01-01..{year}-12-31"
        url = f"https://api.github.com/search/repositories?q={query}"
        try:
            res = requests.get(url).json()
            return res.get('total_count', 0)
        except: return 0

    def run_pipeline(self):
        print(f"🚀 Starting Master Ingestion for {len(self.skills)} skills...")
        results = []
        for skill in self.skills:
            print(f"📡 Syncing: {skill}")
            entry = {'skill': skill}
            for yr, (s, e) in self.years.items():
                entry[f'so_{yr}'] = self.fetch_so(skill, s, e)
                entry[f'gh_{yr}'] = self.fetch_gh(skill, yr)
                # Respecting rate limits is key for 5-API stability
                time.sleep(2) 
            results.append(entry)
        
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        pd.DataFrame(results).to_sql('master_trends', conn, if_exists='replace', index=False)
        conn.close()
        print(f"✅ Success! 21 skills synced to {self.db_path}")

if __name__ == "__main__":
    factory = SkillDataFactory()
    factory.run_pipeline()