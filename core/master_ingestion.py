import requests
import pandas as pd
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────
# Stack Overflow tag mapping:
# Some skill names need to be mapped to their exact SO tag name
# ─────────────────────────────────────────────────────────────
SO_TAG_MAP = {
    'python': 'python', 'javascript': 'javascript', 'java': 'java',
    'c++': 'c%2B%2B', 'c#': 'c%23', 'rust': 'rust', 'go': 'go',
    'typescript': 'typescript', 'swift': 'swift', 'kotlin': 'kotlin',
    'ruby': 'ruby', 'php': 'php', 'scala': 'scala', 'r': 'r',
    'dart': 'dart', 'julia': 'julia', 'objective-c': 'objective-c',
    'perl': 'perl', 'lua': 'lua', 'bash': 'bash',
    'reactjs': 'reactjs', 'angular': 'angular', 'vue.js': 'vue.js',
    'svelte': 'svelte', 'next.js': 'next.js', 'html': 'html',
    'css': 'css', 'tailwind-css': 'tailwind-css', 'bootstrap': 'twitter-bootstrap',
    'jquery': 'jquery', 'node.js': 'node.js', 'django': 'django',
    'flask': 'flask', 'spring': 'spring', 'laravel': 'laravel',
    'fastapi': 'fastapi', 'express': 'express', 'graphql': 'graphql',
    'asp.net': 'asp.net', 'react-native': 'react-native', 'flutter': 'flutter',
    'android': 'android', 'ios': 'ios', 'xamarin': 'xamarin', 'ionic': 'ionic',
    'mysql': 'mysql', 'postgresql': 'postgresql', 'mongodb': 'mongodb',
    'redis': 'redis', 'sqlite': 'sqlite', 'elasticsearch': 'elasticsearch',
    'oracle': 'oracle', 'cassandra': 'cassandra', 'dynamodb': 'amazon-dynamodb',
    'firebase': 'firebase', 'aws': 'amazon-web-services', 'azure': 'azure',
    # Cloud additions
    'vercel': 'vercel', 'netlify': 'netlify', 'pulumi': 'pulumi', 'cloudflare': 'cloudflare',
    # Database additions
    'supabase': 'supabase', 'prisma': 'prisma', 'neo4j': 'neo4j', 'clickhouse': 'clickhouse',
    # Cybersecurity
    'kali-linux': 'kali-linux', 'burp-suite': 'burp-suite', 'wireshark': 'wireshark',
    'splunk': 'splunk', 'ethical-hacking': 'ethical-hacking',
    # Generative AI & LLMs
    'langchain': 'langchain', 'openai-api': 'openai', 'hugging-face': 'huggingface-transformers',
    'llama': 'llama.cpp', 'stable-diffusion': 'stable-diffusion', 'prompt-engineering': 'prompt-engineering',
    # Web3 & Blockchain
    'solidity': 'solidity', 'web3.js': 'web3js', 'hardhat': 'hardhat',
    # Game Dev
    'unity': 'unity3d', 'unreal-engine': 'unreal-engine', 'godot': 'godot',
    'google-cloud-platform': 'google-cloud-platform', 'docker': 'docker',
    'kubernetes': 'kubernetes', 'terraform': 'terraform', 'linux': 'linux',
    'git': 'git', 'jenkins': 'jenkins', 'github-actions': 'github-actions',
    'ansible': 'ansible', 'nginx': 'nginx', 'pandas': 'pandas',
    'numpy': 'numpy', 'tensorflow': 'tensorflow', 'pytorch': 'pytorch',
    'scikit-learn': 'scikit-learn', 'hadoop': 'hadoop', 'apache-spark': 'apache-spark',
    'apache-kafka': 'apache-kafka', 'snowflake': 'snowflake', 'airflow': 'airflow',
    'cypress': 'cypress', 'selenium': 'selenium', 'jest': 'jestjs',
    'pytest': 'pytest', 'mocha': 'mocha', 'figma': 'figma', 'jira': 'jira'
}

PYPI_PACKAGES = {
    'python', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn',
    'flask', 'django', 'fastapi', 'airflow', 'pytest'
}
PYPI_MAP = {
    'python': None, 'pandas': 'pandas', 'numpy': 'numpy',
    'tensorflow': 'tensorflow', 'pytorch': 'torch', 'scikit-learn': 'scikit-learn',
    'flask': 'flask', 'django': 'django', 'fastapi': 'fastapi',
    'airflow': 'apache-airflow', 'pytest': 'pytest'
}

NPM_PACKAGES = {
    'javascript', 'typescript', 'reactjs', 'angular', 'vue.js', 'svelte',
    'next.js', 'jquery', 'node.js', 'express', 'jest', 'mocha', 'cypress'
}
NPM_MAP = {
    'javascript': None, 'typescript': 'typescript', 'reactjs': 'react',
    'angular': '@angular/core', 'vue.js': 'vue', 'svelte': 'svelte',
    'next.js': 'next', 'jquery': 'jquery', 'node.js': None,
    'express': 'express', 'jest': 'jest', 'mocha': 'mocha', 'cypress': 'cypress'
}

YEARS = {
    '2020': (1577836800, 1609459199),
    '2021': (1609459200, 1640995199),
    '2022': (1640995200, 1672531199),
    '2023': (1672531200, 1704067199),
    '2024': (1704067200, 1735689599),
}

def fetch_so_tag(tag, fromdate, todate):
    """Fetch total Stack Overflow questions for a tag in a year range."""
    url = (
        f"https://api.stackexchange.com/2.3/questions"
        f"?fromdate={fromdate}&todate={todate}"
        f"&tagged={tag}&site=stackoverflow&filter=total"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        return resp.get('total', 0)
    except:
        return 0

def fetch_pypi_monthly(package):
    """Fetch last 6 months total downloads from PyPI Stats."""
    try:
        url = f"https://pypistats.org/api/packages/{package}/recent"
        resp = requests.get(url, timeout=10).json()
        return resp.get('data', {}).get('last_month', 0)
    except:
        return 0

def fetch_npm_monthly(package):
    """Fetch last month downloads from npm registry."""
    try:
        package_encoded = package.replace('/', '%2F')
        url = f"https://api.npmjs.org/downloads/point/last-month/{package_encoded}"
        resp = requests.get(url, timeout=10).json()
        return resp.get('downloads', 0)
    except:
        return 0

def run_real_ingestion():
    """
    Main pipeline: fetch real 5-year StackOverflow question data for all 84 skills.
    Supplements with PyPI/npm download signals for ecosystem-specific skills.
    """
    skills = list(SO_TAG_MAP.keys())
    print(f"Starting Real Data Ingestion for {len(skills)} skills...")
    print("Source: Stack Overflow Tag API + PyPI Stats + npm Registry")
    print("-" * 60)

    records = []

    for skill in skills:
        tag = SO_TAG_MAP[skill]
        print(f"  Fetching: {skill}")
        row = {'Skill': skill}

        for year, (start, end) in YEARS.items():
            count = fetch_so_tag(tag, start, end)
            row[f'SO_{year}'] = count
            # Stack Overflow API has quota; add tiny delay to be polite
            time.sleep(0.3)

        # Supplement: PyPI boost for Python-ecosystem packages
        if skill in PYPI_PACKAGES and PYPI_MAP.get(skill):
            row['PyPI_Monthly'] = fetch_pypi_monthly(PYPI_MAP[skill])
        else:
            row['PyPI_Monthly'] = 0

        # Supplement: npm boost for JS-ecosystem packages
        if skill in NPM_PACKAGES and NPM_MAP.get(skill):
            row['NPM_Monthly'] = fetch_npm_monthly(NPM_MAP[skill])
        else:
            row['NPM_Monthly'] = 0

        records.append(row)

    df = pd.DataFrame(records)

    # Compute a unified Job_Demand: latest year SO + 10% of downloads
    df['Job_Demand'] = df['SO_2024'] + (df['PyPI_Monthly'] * 0.0001).astype(int) + (df['NPM_Monthly'] * 0.0001).astype(int)

    # Build unified_dataset format (skill x year rows) for ARIMA
    rows = []
    for _, r in df.iterrows():
        for year in YEARS.keys():
            rows.append({
                'Skill': r['Skill'],
                'Year': int(year),
                'Job_Demand': r[f'SO_{year}']
            })

    unified_df = pd.DataFrame(rows)

    if not os.path.exists('data'):
        os.makedirs('data')

    unified_df.to_csv('data/unified_dataset.csv', index=False)
    df.to_csv('data/raw_real_data.csv', index=False)

    print("-" * 60)
    print(f"Done! Saved {len(skills)} skills x 5 years = {len(unified_df)} records.")
    print("Output: data/unified_dataset.csv (for ARIMA) + data/raw_real_data.csv (raw)")

if __name__ == "__main__":
    run_real_ingestion()