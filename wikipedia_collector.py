import wikipediaapi
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import json
import time

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class WikipediaCollector:
    def __init__(self, language='en'):
        self.wiki = wikipediaapi.Wikipedia(language)
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean Wikipedia text content"""
        # Remove Wikipedia markup
        text = re.sub(r'==.*?==', '', text)  # Remove section headers
        text = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', text)  # Remove internal links but keep text
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)  # Remove remaining internal links
        text = re.sub(r'\[.*?\]', '', text)  # Remove external links
        text = re.sub(r'\{.*?\}', '', text)  # Remove templates
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'\([^)]*\)', '', text)  # Remove parentheses content
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)  # Keep only alphanumeric, spaces, and punctuation
        
        # Normalize whitespace
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_page_content(self, page_title):
        """Get cleaned content from a Wikipedia page"""
        try:
            page = self.wiki.page(page_title)
            if page.exists():
                content = page.text
                cleaned_content = self.clean_text(content)
                sentences = sent_tokenize(cleaned_content)
                return [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
            else:
                print(f"Page '{page_title}' not found")
                return []
        except Exception as e:
            print(f"Error fetching page '{page_title}': {e}")
            return []
    
    def collect_corpus(self, page_titles, delay=1):
        """Collect sentences from multiple Wikipedia pages"""
        corpus = []
        
        for i, title in enumerate(page_titles):
            print(f"Collecting from page {i+1}/{len(page_titles)}: {title}")
            sentences = self.get_page_content(title)
            corpus.extend(sentences)
            
            # Add delay to be respectful to Wikipedia's servers
            if i < len(page_titles) - 1:
                time.sleep(delay)
        
        return corpus
    
    def save_corpus(self, corpus, filename='wikipedia_corpus.json'):
        """Save corpus to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f"Corpus saved to {filename} with {len(corpus)} sentences")
    
    def load_corpus(self, filename='wikipedia_corpus.json'):
        """Load corpus from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            print(f"Loaded corpus from {filename} with {len(corpus)} sentences")
            return corpus
        except FileNotFoundError:
            print(f"Corpus file {filename} not found")
            return []

# Example usage
if __name__ == "__main__":
    # Sample Wikipedia pages for different topics
    pages = [
        "Artificial intelligence",
        "Machine learning", 
        "Natural language processing",
        "Computer science",
        "Mathematics",
        "Physics",
        "Biology",
        "Chemistry",
        "History",
        "Geography"
    ]
    
    collector = WikipediaCollector()
    
    # Collect corpus (uncomment to collect new data)
    # corpus = collector.collect_corpus(pages)
    # collector.save_corpus(corpus)
    
    # Or load existing corpus
    corpus = collector.load_corpus()
    
    if corpus:
        print(f"Sample sentences from corpus:")
        for i, sentence in enumerate(corpus[:5]):
            print(f"{i+1}. {sentence}") 