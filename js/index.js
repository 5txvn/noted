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
      'because', 'if', 'when', 'where', 'while', 'although', 'though', 'until',
      // ✅ Units and numeric suffixes
      'inch', 'inches', 'feet', 'foot', 'ft', 'centimeter', 'centimeters', 'cm',
      'meter', 'meters', 'm', 'mm', 'kilogram', 'kg', 'pound', 'pounds', 'lbs', 'oz',
      'degree', 'degrees', '°f', '°c'
    ];
    extraStopwords.forEach(w => this.stopWords.add(w));

    this.fetchedPagesFile = 'fetched_pages.json';
    this.skippedPagesFile = 'skipped_low_views.json';

    this.fetchedPages = this.loadList(this.fetchedPagesFile);
    this.skippedPages = this.loadList(this.skippedPagesFile);
  }

  loadList(filename) {
    if (!fs.existsSync(filename)) return new Set();
    return new Set(JSON.parse(fs.readFileSync(filename, 'utf-8')));
  }

  saveList(filename, dataSet) {
    fs.writeFileSync(filename, JSON.stringify(Array.from(dataSet), null, 2));
  }

  splitSentences(text) {
    return text.split(/(?<=[.?!])\s+(?=[A-Z])/);
  }

  cleanSentence(sentence) {
    sentence = sentence.replace(/\([^)]*\)/g, ' ');
    sentence = sentence.replace(/\{[^}]*\}/g, ' ');
    sentence = sentence.replace(/\[[^\]]*\]/g, ' ');
    sentence = sentence.replace(/<[^>]*>/g, ' ');
    sentence = sentence.replace(/==.*?==/g, ' ');
    sentence = sentence.replace(/\[\[(?:.*?\|)?(.*?)\]\]/g, '$1');
    sentence = sentence.replace(/-/g, ' ');
    sentence = sentence.replace(/\s+/g, ' ').trim();

    let words = sentence.split(' ').map(w => w.toLowerCase());

    words = words.filter(word => {
      if (!word) return false;
      if (word.length === 1) return false;                  // ✅ Skip 1-letter words
      if (/^\d+$/.test(word)) return false;                 // ✅ Skip numbers
      if (/[^a-z]/.test(word)) return false;                // Skip non-alpha
      if (this.stopWords.has(word)) return false;           // Skip stopwords (incl. units)
      return true;
    });

    if (words.length < 4) return null;
    return words.join(' ');
  }

  cleanTextToSentences(text) {
    const rawSentences = this.splitSentences(text);
    return rawSentences
      .map(sentence => this.cleanSentence(sentence))
      .filter(Boolean); // remove nulls
  }

  async getPageViews(title) {
    const today = new Date();
    const end = today.toISOString().split('T')[0].replace(/-/g, '');
    const startDate = new Date(today.setDate(today.getDate() - 30));
    const start = startDate.toISOString().split('T')[0].replace(/-/g, '');

    const url = `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/${this.language}.wikipedia.org/all-access/user/${encodeURIComponent(title)}/daily/${start}/${end}`;
    try {
      const res = await axios.get(url);
      return res.data.items.reduce((sum, entry) => sum + entry.views, 0);
    } catch {
      return 0;
    }
  }

  async getPageContent(title) {
    const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&titles=${encodeURIComponent(title)}&redirects=1`;
    try {
      const res = await axios.get(url);
      const pages = res.data.query.pages;
      const page = pages[Object.keys(pages)[0]];
      return page?.extract ? this.cleanTextToSentences(page.extract) : [];
    } catch {
      return [];
    }
  }

  async getRandomPages(limit = 100) {
    const titles = new Set();
    try {
      while (titles.size < limit) {
        const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=${Math.min(10, limit - titles.size)}&format=json&rnnamespace=0`;
        const res = await axios.get(url);
        res.data.query.random.forEach(page => {
          if (!this.fetchedPages.has(page.title)) {
            titles.add(page.title);
          }
        });
      }
    } catch (err) {
      console.error("Error fetching random pages:", err.message);
    }
    return Array.from(titles);
  }

  async getPagesFromCategory(category, limit = 50) {
    const titles = new Set();
    let cmcontinue = null;

    while (titles.size < limit) {
      const url = `https://${this.language}.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:${encodeURIComponent(category)}&cmlimit=50&format=json${cmcontinue ? `&cmcontinue=${cmcontinue}` : ''}`;
      const res = await axios.get(url);
      const pages = res.data.query.categorymembers;

      for (const page of pages) {
        if (page.ns === 0 && !this.fetchedPages.has(page.title)) {  // ns=0 means it's a main/article page
          titles.add(page.title);
          if (titles.size >= limit) break;
        }
      }

      if (res.data.continue && res.data.continue.cmcontinue) {
        cmcontinue = res.data.continue.cmcontinue;
      } else {
        break;
      }
    }

    return Array.from(titles);
  }

  async collectCorpus(targetArticleCount = 10, viewThreshold = 1000, delay = 250) {
    const corpus = [];
    let collectedArticles = 0;

    while (collectedArticles < targetArticleCount) {
      const batch = await this.getRandomPages((targetArticleCount - collectedArticles) * 3);

      for (const title of batch) {
        if (this.fetchedPages.has(title)) continue;

        const views = await this.getPageViews(title);
        if (views < viewThreshold) {
          this.skippedPages.add(title);
          continue;
        }

        const cleanedSentences = await this.getPageContent(title);
        if (cleanedSentences.length > 0) {
          corpus.push(...cleanedSentences);
          this.fetchedPages.add(title);
          collectedArticles++;
          console.log(`✅ ${collectedArticles}/${targetArticleCount}: ${title} (${views} views)`);
        } else {
          console.log(`⚠️ Skipped (empty content): ${title}`);
          this.skippedPages.add(title);
        }

        if (collectedArticles >= targetArticleCount) break;
        await new Promise(r => setTimeout(r, delay));
      }
    }

    this.saveList(this.fetchedPagesFile, this.fetchedPages);
    this.saveList(this.skippedPagesFile, this.skippedPages);
    return corpus;
  }

  saveCorpus(corpus, filename = 'wikipedia_corpus.json') {
    fs.writeFileSync(filename, JSON.stringify(corpus, null, 2), 'utf-8');
    console.log(`✅ Corpus saved to ${filename} with ${corpus.length} cleaned sentences.`);
  }

  loadCorpus(filename = 'wikipedia_corpus.json') {
    return fs.existsSync(filename)
      ? JSON.parse(fs.readFileSync(filename, 'utf-8'))
      : [];
  }
}

// ✅ Example usage
(async () => {
  const collector = new WikipediaCollector();
  const targetArticles = 10000;
  const pageViewThreshold = 1000;
  const delay = 250;

  console.log(`📦 Collecting ${targetArticles} full articles...`);
  const corpus = await collector.collectCorpus(targetArticles, pageViewThreshold, delay);
  collector.saveCorpus(corpus);
})();
//i can hear music