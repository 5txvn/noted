// WikipediaCollector.js
const fs = require('fs');
const axios = require('axios');
const natural = require('natural');
const stopword = require('stopword');

console.log(stopword.en)

class WikipediaCollector {
  constructor(language = 'en') {
    this.language = language;
    this.stopWords = new Set(stopword.en);
    this.tokenizer = new natural.SentenceTokenizer();
  }

  cleanText(text) {
    // Remove Wikipedia markup
    text = text.replace(/==.*?==/g, ''); // Section headers
    text = text.replace(/\[\[(.*?\|)?(.*?)\]\]/g, '$2'); // Internal links
    text = text.replace(/\[.*?\]/g, ''); // External links
    text = text.replace(/\{.*?\}/g, ''); // Templates
    text = text.replace(/<.*?>/g, ''); // HTML tags
    text = text.replace(/\([^)]*\)/g, ''); // Parentheses
    text = text.replace(/[^\w\s\.,!?]/g, ''); // Keep alphanumeric and punctuation

    // Normalize whitespace
    text = text.replace(/[\n\r\t]+/g, ' ');
    text = text.replace(/\s+/g, ' ').trim();

    return text;
  }

  async getPageContent(title) {
    const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&titles=${encodeURIComponent(title)}&redirects=1`;
    try {
      const res = await axios.get(url);
      const pages = res.data.query.pages;
      const page = pages[Object.keys(pages)[0]];
      if (!page || !page.extract) {
        console.log(`Page '${title}' not found.`);
        return [];
      }
      const cleaned = this.cleanText(page.extract);
      const sentences = this.tokenizer.tokenize(cleaned);
      return sentences.filter(s => s.length > 20);
    } catch (err) {
      console.error(`Error fetching page '${title}':`, err.message);
      return [];
    }
  }

  async collectCorpus(pageTitles, delay = 1000) {
    const corpus = [];
    for (let i = 0; i < pageTitles.length; i++) {
      const title = pageTitles[i];
      console.log(`Collecting from page ${i + 1}/${pageTitles.length}: ${title}`);
      const sentences = await this.getPageContent(title);
      corpus.push(...sentences);
      if (i < pageTitles.length - 1) await new Promise(r => setTimeout(r, delay));
    }
    return corpus;
  }

  saveCorpus(corpus, filename = 'wikipedia_corpus.json') {
    fs.writeFileSync(filename, JSON.stringify(corpus, null, 2), 'utf-8');
    console.log(`Corpus saved to ${filename} with ${corpus.length} sentences.`);
  }

  loadCorpus(filename = 'wikipedia_corpus.json') {
    if (!fs.existsSync(filename)) {
      console.log(`Corpus file ${filename} not found.`);
      return [];
    }
    const data = fs.readFileSync(filename, 'utf-8');
    const corpus = JSON.parse(data);
    console.log(`Loaded corpus from ${filename} with ${corpus.length} sentences.`);
    return corpus;
  }
}

// Example usage
(async () => {
  const pages = [
    "Artificial intelligence",
    "Machine learning"
  ];

  const collector = new WikipediaCollector();

  // Uncomment to collect fresh data
  const corpus = await collector.collectCorpus(pages);
  collector.saveCorpus(corpus);

  // Load existing
//   const corpus = collector.loadCorpus();
//   if (corpus.length) {
//     console.log("Sample sentences from corpus:");
//     for (let i = 0; i < 5; i++) {
//       console.log(`${i + 1}. ${corpus[i]}`);
//     }
//   }
})();
