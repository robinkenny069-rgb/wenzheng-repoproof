"""Small transparent BM25 baseline for the same English retrieval corpus.

Uses the same token boundaries as cli.terms, retaining document frequencies.
No stemming, learned weights, stopword list, reference labels or test tuning.
This is a local implementation of the formula, not the Lucene executable.
"""
from collections import Counter
import math
import re


def tokens(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return re.findall(r'[a-z]+|[0-9]+', text.lower().replace('_', ' '))


class BM25Index:
    def __init__(self, documents, *, k1=1.2, b=0.75):
        if not math.isfinite(k1) or k1 < 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError('BM25 needs finite k1 >= 0 and b in [0, 1]')
        self.k1, self.b = k1, b
        self.counts = [Counter(tokens(document)) for document in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths)/len(self.lengths) if self.lengths else 0
        self.document_frequency = Counter(term for count in self.counts for term in count)

    def rank(self, query):
        terms = sorted(set(tokens(query)))
        scores = []
        n = len(self.counts)
        for index, count in enumerate(self.counts):
            normalization = self.k1 * (1-self.b+self.b*self.lengths[index]/self.average_length) if self.average_length else 0
            score = 0.0
            for term in terms:
                frequency = count[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log1p((n-df+0.5)/(df+0.5))
                score += idf * frequency*(self.k1+1)/(frequency+normalization)
            scores.append((index, score))
        # Keep zero-score documents, with the same deterministic tie rule as the other methods.
        return sorted(scores, key=lambda row: (-row[1], row[0]))

    def specification(self):
        return {'method':'BM25', 'implementation':'local formula; not Lucene runtime',
                'k1':self.k1, 'b':self.b, 'query_term_weight':'unique query terms',
                'tokenization':'same ASCII token boundaries as lexical Jaccard, with document term frequencies',
                'idf':'ln(1 + (N - df + 0.5)/(df + 0.5))',
                'tf':'tf * (k1 + 1) / (tf + k1*(1-b+b*dl/avgdl))',
                'parameter_reference':'https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html',
                'reference_note':'Parameters and positive IDF follow the linked public documentation; exact floating scores are not claimed equal to Lucene norms.',
                'corpus_documents':len(self.counts), 'average_document_length':self.average_length}
