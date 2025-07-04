const fs = require('fs');
const bz2 = require('unbzip2-stream');
const sax = require('sax');
const stopword = require('stopword');

class WikipediaCollector {
  constructor(language = 'simple') {
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
      'inch', 'inches', 'feet', 'foot', 'ft', 'centimeter', 'centimeters', 'cm',
      'meter', 'meters', 'm', 'mm', 'kilogram', 'kg', 'pound', 'pounds', 'lbs', 'oz',
      'degree', 'degrees', '°f', '°c'
    ];
    extraStopwords.forEach(w => this.stopWords.add(w));
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
      if (word.length === 1) return false;
      if (/^\d+$/.test(word)) return false;
      if (/[^a-z]/.test(word)) return false;
      if (this.stopWords.has(word)) return false;
      return true;
    });

    if (words.length < 4) return null;
    return words.join(' ');
  }

  cleanTextToSentences(text) {
    const rawSentences = this.splitSentences(text);
    return rawSentences
      .map(sentence => this.cleanSentence(sentence))
      .filter(Boolean);
  }

  shouldSkipArticle(title, text) {
    if (title.includes('(disambiguation)') || 
        title.includes('(redirect)') ||
        title.startsWith('Template:') ||
        title.startsWith('Category:') ||
        title.startsWith('File:') ||
        title.startsWith('User:') ||
        title.startsWith('Talk:') ||
        title.startsWith('Wikipedia:') ||
        title.startsWith('Help:') ||
        title.startsWith('Portal:') ||
        title.startsWith('MediaWiki:') ||
        title.startsWith('Special:') ||
        title.startsWith('Module:')) {
      return true;
    }

    if (text.length < 100) {
      return true;
    }

    return false;
  }

  async collectFromDump(dumpPath, targetSentenceCount = 100000) {
    console.log(`📦 Reading and decompressing ${dumpPath}...`);

    const stream = fs.createReadStream(dumpPath);
    const decompressed = stream.pipe(bz2());  // Pipe through unbzip2-stream

    const parser = sax.createStream(true, { trim: true });

    const corpus = [];
    let articleCount = 0;
    let sentenceCount = 0;
    let skippedCount = 0;

    let currentPage = {};
    let currentElement = '';
    let currentText = '';
    let inRevision = false;
    let inText = false;

    return new Promise((resolve, reject) => {
      parser.on('opentag', (node) => {
        currentElement = node.name;

        if (node.name === 'page') {
          currentPage = {};
        } else if (node.name === 'revision') {
          inRevision = true;
        } else if (node.name === 'text' && inRevision) {
          inText = true;
          currentText = '';
        }
      });

      parser.on('text', (text) => {
        if (inText) {
          currentText += text;
        } else if (currentElement === 'title') {
          currentPage.title = text;
        }
      });

      parser.on('closetag', (nodeName) => {
        if (nodeName === 'text' && inText) {
          inText = false;
          currentPage.text = currentText;
        } else if (nodeName === 'revision') {
          inRevision = false;
        } else if (nodeName === 'page') {
          try {
            const title = currentPage.title || '';
            const text = currentPage.text || '';

            if (this.shouldSkipArticle(title, text)) {
              skippedCount++;
              return;
            }

            const cleanedSentences = this.cleanTextToSentences(text);

            if (cleanedSentences.length > 0) {
              corpus.push(...cleanedSentences);
              articleCount++;
              sentenceCount += cleanedSentences.length;

              if (articleCount % 100 === 0) {
                console.log(`✅ Processed ${articleCount} articles | ${sentenceCount} sentences | Skipped: ${skippedCount}`);
              }

              if (sentenceCount >= targetSentenceCount) {
                console.log(`🎯 Reached target of ${targetSentenceCount} sentences, stopping...`);
                parser.end();  // Stop parsing early if you want
                return;
              }
            }
          } catch (error) {
            console.error(`Error processing article: ${error.message}`);
          }

          currentPage = {};
        }
      });

      parser.on('end', () => {
        this.saveCorpus(corpus);
        console.log(`✅ Finished. Saved ${sentenceCount} sentences from ${articleCount} articles.`);
        console.log(`📊 Skipped ${skippedCount} articles.`);
        resolve(corpus);
      });

      parser.on('error', (err) => {
        console.error('XML parsing error:', err);
        reject(err);
      });

      decompressed.pipe(parser);  // Pipe decompressed stream into SAX parser
    });
  }

  saveCorpus(corpus, filename = 'corpus.json') {
    fs.writeFileSync(filename, JSON.stringify(corpus, null, 2), 'utf-8');
    console.log(`✅ Corpus saved to ${filename} with ${corpus.length} cleaned sentences.`);
  }

  loadCorpus(filename = 'corpus.json') {
    return fs.existsSync(filename)
      ? JSON.parse(fs.readFileSync(filename, 'utf-8'))
      : [];
  }
}

// ✅ Entry point
(async () => {
  const collector = new WikipediaCollector();
  const dumpPath = 'simplewiki-latest-pages-articles.xml.bz2';
  const targetSentences = 10000000;

  console.log(`📦 Processing Simple English Wikipedia dump...`);
  console.log(`🎯 Target: ${targetSentences.toLocaleString()} sentences`);
  
  try {
    await collector.collectFromDump(dumpPath, targetSentences);
  } catch (error) {
    console.error('❌ Error processing dump:', error.message);
    process.exit(1);
  }
})();
