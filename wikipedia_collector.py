import os
import json
import time
import re
import requests
from typing import List, Set
import nltk

# Setup NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class WikipediaCollector:
    def __init__(self, language='en'):
        self.language = language
        self.stopwords = self._load_stopwords()

        self.fetched_pages_file = 'fetched_pages.json'
        self.skipped_pages_file = 'skipped_low_views.json'

        self.fetched_pages = self._load_set(self.fetched_pages_file)
        self.skipped_pages = self._load_set(self.skipped_pages_file)

    def _load_set(self, filename: str) -> Set[str]:
        if not os.path.exists(filename):
            return set()
        with open(filename, 'r', encoding='utf-8') as f:
            return set(json.load(f))

    def _save_set(self, filename: str, data: Set[str]):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(list(data), f, indent=2)

    def _load_stopwords(self) -> Set[str]:
        sw = set(stopwords.words('english'))
        sw.update([
            'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours',
            'his', 'her', 'hers', 'their', 'theirs', 'our', 'ours',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'have', 'has', 'had',
            'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
            'also', 'just', 'like', 'so', 'very', 'such', 'one', 'two', 'first', 'second',
            'new', 'many', 'much', 'some', 'any', 'all', 'each', 'every', 'no', 'not',
            'however', 'than', 'then', 'too', 'more', 'most', 'other', 'others',
            'because', 'if', 'when', 'where', 'while', 'although', 'though', 'until'
        ])
        return sw

    def _clean_text(self, text: str) -> List[str]:
        text = re.sub(r'\([^)]*\)', ' ', text)
        text = re.sub(r'\{[^}]*\}', ' ', text)
        text = re.sub(r'\[[^\]]]*\]', ' ', text)
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'==.*?==', ' ', text)
        text = re.sub(r'\[\[(?:.*?\|)?(.*?)\]\]', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            cleaned_tokens = [
                word.lower() for word in tokens
                if word.isalpha() and word.lower() not in self.stopwords
            ]
            if cleaned_tokens:
                cleaned_sentences.append(' '.join(cleaned_tokens))
        return cleaned_sentences

    def get_page_views(self, title: str) -> int:
        today = time.strftime("%Y%m%d")
        start = time.strftime("%Y%m%d", time.gmtime(time.time() - 30 * 86400))
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{self.language}.wikipedia.org/all-access/user/{requests.utils.quote(title)}/daily/{start}/{today}"
        try:
            response = requests.get(url)
            data = response.json()
            return sum(item['views'] for item in data.get('items', []))
        except:
            return 0

    def get_page_content(self, title: str) -> List[str]:
        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'prop': 'extracts',
            'explaintext': True,
            'format': 'json',
            'titles': title,
            'redirects': 1
        }
        try:
            res = requests.get(url, params=params).json()
            pages = res['query']['pages']
            page = next(iter(pages.values()))
            if not page or 'extract' not in page:
                return []
            return self._clean_text(page['extract'])
        except:
            return []

    def get_random_titles(self, limit: int) -> List[str]:
        titles = set()
        while len(titles) < limit:
            url = f"https://{self.language}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'random',
                'rnlimit': min(10, limit - len(titles)),
                'format': 'json',
                'rnnamespace': 0
            }
            try:
                res = requests.get(url, params=params).json()
                for page in res['query']['random']:
                    if page['title'] not in self.fetched_pages:
                        titles.add(page['title'])
            except:
                continue
        return list(titles)

    def collect_corpus(self, target_size=1000, min_views=2500, delay=0.25) -> List[str]:
        corpus = []
        while len(corpus) < target_size:
            print(target_size)
            titles = self.get_random_titles(30)
            for title in titles:
                if title in self.fetched_pages:
                    continue
                views = self.get_page_views(title)
                if views < min_views:
                    self.skipped_pages.add(title)
                    continue
                content = self.get_page_content(title)
                if content:
                    corpus.extend(content)
                    self.fetched_pages.add(title)
                    print(f"✅ {len(corpus)} / {target_size} : {title} ({views} views)")
                else:
                    print(f"⚠️ Skipped (empty): {title}")
                    self.skipped_pages.add(title)
                if len(corpus) >= target_size:
                    break
                time.sleep(delay)
        self._save_set(self.fetched_pages_file, self.fetched_pages)
        self._save_set(self.skipped_pages_file, self.skipped_pages)
        return corpus

    def save_corpus(self, corpus: List[str], filename='wikipedia_corpus.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2)
        print(f"✅ Saved {len(corpus)} sentences to {filename}")

    def load_corpus(self, filename='wikipedia_corpus.json') -> List[str]:
        if not os.path.exists(filename):
            return []
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    collector = WikipediaCollector()

    target_size = 100  # total number of clean sentences
    min_views = 5000    # minimum 30-day pageviews
    delay = 0.25        # seconds between requests

    print(f"📦 Collecting {target_size} cleaned sentences from Wikipedia...")
    corpus = collector.collect_corpus(target_size, min_views, delay)
    collector.save_corpus(corpus)
