const fs = require('fs');
const axios = require('axios');
const stopword = require('stopword');

class WikipediaCollector {
  constructor(language = 'en') {
    this.language = language;
    this.stopWords = new Set([...stopword.eng]);

    const extraStopwords = [
      'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours',
      'his', 'her', 'hers', 'their', 'theirs', 'our', 'ours',
      'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'do', 'does', 'did', 'have', 'has', 'had',
      'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
      'also', 'just', 'like', 'so', 'very', 'such', 'one', 'two', 'first', 'second',
      'new', 'many', 'much', 'some', 'any', 'all', 'each', 'every', 'no', 'not',
      'however', 'than', 'then', 'too', 'more', 'most', 'other', 'others',
      'because', 'if', 'when', 'where', 'while', 'although', 'though', 'until'
    ];
    extraStopwords.forEach(w => this.stopWords.add(w));

    this.fetchedPagesFile = 'fetched_pages.json';
    this.skippedPagesFile = 'skipped_low_views.json';

    this.fetchedPages = this.loadList(this.fetchedPagesFile);
    this.skippedPages = this.loadList(this.skippedPagesFile);
  }

  loadList(filename) {
    if (!fs.existsSync(filename)) return new Set();
    const data = fs.readFileSync(filename, 'utf-8');
    return new Set(JSON.parse(data));
  }

  saveList(filename, dataSet) {
    fs.writeFileSync(filename, JSON.stringify(Array.from(dataSet), null, 2));
  }

  cleanText(text) {
    text = text.replace(/\([^)]*\)/g, ' ');
    text = text.replace(/\{[^}]*\}/g, ' ');
    text = text.replace(/\[[^\]]*\]/g, ' ');
    text = text.replace(/<[^>]*>/g, ' ');
    text = text.replace(/==.*?==/g, ' ');
    text = text.replace(/\[\[(?:.*?\|)?(.*?)\]\]/g, '$1');
    text = text.replace(/\s+/g, ' ').trim();

    let words = text.split(' ').map(w => w.toLowerCase());

    words = words.filter(word => {
      if (!word) return false;
      if (/\d/.test(word)) return false;
      if (/[^a-z]/.test(word)) return false;
      if (this.stopWords.has(word)) return false;
      return true;
    });

    return words.join(' ');
  }

  async getPageViews(title) {
    const today = new Date();
    const end = today.toISOString().split('T')[0].replace(/-/g, '');
    const startDate = new Date(today.setDate(today.getDate() - 30));
    const start = startDate.toISOString().split('T')[0].replace(/-/g, '');

    const url = `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/${this.language}.wikipedia.org/all-access/user/${encodeURIComponent(title)}/daily/${start}/${end}`;
    try {
      const res = await axios.get(url);
      const totalViews = res.data.items.reduce((sum, entry) => sum + entry.views, 0);
      return totalViews;
    } catch (err) {
      return 0; // treat failures as zero views
    }
  }

  async getPageContent(title) {
    const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&titles=${encodeURIComponent(title)}&redirects=1`;
    try {
      const res = await axios.get(url);
      const pages = res.data.query.pages;
      const page = pages[Object.keys(pages)[0]];
      if (!page || !page.extract) {
        return '';
      }
      const cleaned = this.cleanText(page.extract);
      return cleaned;
    } catch (err) {
      return '';
    }
  }

  async getRandomPages(limit = 100) {
    const titles = new Set();
    try {
      while (titles.size < limit) {
        const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=${Math.min(10, limit - titles.size)}&format=json&rnnamespace=0`;
        const res = await axios.get(url);
        res.data.query.random.forEach(page => {
          if (!this.fetchedPages.has(page.title) && !titles.has(page.title)) {
            titles.add(page.title);
          }
        });
      }
    } catch (err) {
      console.error("Error fetching random pages:", err.message);
    }
    return Array.from(titles);
  }

  async collectCorpus(targetCorpusSize = 100, viewThreshold = 1000, delay = 250) {
    const corpus = [];
    while (corpus.length < targetCorpusSize) {
      const batchSize = Math.min(10, targetCorpusSize - corpus.length);
      const batch = await this.getRandomPages(batchSize * 3); // fetch extra in case some fail

      for (const title of batch) {
        if (this.fetchedPages.has(title)) continue;

        const views = await this.getPageViews(title);
        if (views < viewThreshold) {
          this.skippedPages.add(title);
          continue;
        }

        const cleaned = await this.getPageContent(title);
        if (cleaned && cleaned.length > 0) {
          corpus.push(cleaned);
          this.fetchedPages.add(title);
          console.log(`✅ ${corpus.length}/${targetCorpusSize}: ${title} (${views} views)`);
        } else {
          console.log(`⚠️ Skipped (empty content): ${title}`);
          this.skippedPages.add(title);
        }

        if (corpus.length >= targetCorpusSize) break;

        await new Promise(r => setTimeout(r, delay));
      }
    }

    this.saveList(this.fetchedPagesFile, this.fetchedPages);
    this.saveList(this.skippedPagesFile, this.skippedPages);

    return corpus;
  }

  saveCorpus(corpus, filename = 'wikipedia_corpus.json') {
    fs.writeFileSync(filename, JSON.stringify(corpus, null, 2), 'utf-8');
    console.log(`✅ Corpus saved to ${filename} with ${corpus.length} entries.`);
  }

  loadCorpus(filename = 'wikipedia_corpus.json') {
    if (!fs.existsSync(filename)) {
      return [];
    }
    const data = fs.readFileSync(filename, 'utf-8');
    const corpus = JSON.parse(data);
    return corpus;
  }
}

// ✅ Example usage
(async () => {
  const collector = new WikipediaCollector();

  const targetCorpusSize = 1000;   // Number of valid entries in the final corpus
  const pageViewThreshold = 2500; // Minimum views over past 30 days to accept a page
  const delay = 250;              // Milliseconds between requests

  console.log(`📦 Starting collection of ${targetCorpusSize} entries with min ${pageViewThreshold} views...`);

  const corpus = await collector.collectCorpus(targetCorpusSize, pageViewThreshold, delay);
  collector.saveCorpus(corpus);
})();
