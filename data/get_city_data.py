from dotenv import load_dotenv

load_dotenv('.env')

import json
import os
from pathlib import Path
import requests

"""
GLOBALS
"""
TARGET_PATH = os.environ.get('TARGET_PATH')


if __name__ == '__main__':
    wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston", "New York City"]

    results = []
    for title in wiki_titles:
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                # 'exintro': True,
                'explaintext': True,
            }
        ).json()
        page = next(iter(response['query']['pages'].values()))
        wiki_text = page['extract']
        data_path = Path(TARGET_PATH)
        if not data_path.exists():
            Path.mkdir(data_path)

        results.append({
            'title': page['title'],
            'source': f'https://en.wikipedia.org/wiki/{page["title"]}',
            'page_content': wiki_text
        })

    data_path = Path(TARGET_PATH)
    if not data_path.exists():
        Path.mkdir(data_path)
    with open(data_path / f"cities.json", 'w') as f:
        json.dump(results, f)

    print('DONE')