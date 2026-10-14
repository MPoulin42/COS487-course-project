import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    tokens = re.findall(r'\b[a-z]+\b', clean_text.lower())
    return [token for token in tokens if token not in STOPWORDS]
class tfidf:
    def __init__(self, index_file_path= "Study.json"):
        self.index_file_path = index_file_path
        self.vocab = []
        self.word_to_idx = {}
        self.doc_vectors = {}
        self.documents = {}
        self.df = {}
        self.N = 0
        self.doc_lengths = {}

    def index_file(self, data_path, force_reindex=False):
        """Index raw documents by executing the modular VSM pipeline."""
        if os.path.exists(self.index_file_path) and not force_reindex:
            print(f"Index file '{self.index_file_path}' found. Loading from disk...")
            self.load_index()
            return

        print(f"Indexing source file '{data_path}'...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.N = len(data)
        # docid as key, dict "term: frequency" as value
        doc_term_frequencies = {}

        for doc in data:
            doc_id = doc["Id"]
            # For results representation
            self.documents[doc_id] = doc
            tokens = tokenize_vsm(doc["Text"])
            # get term/token frequency (just in the current doc/answer)
            counts = Counter(tokens)
            doc_term_frequencies[doc_id] = counts

        # Calculates document frequency for each term
        self.calculate_document_frequencies(doc_term_frequencies)
        # Build vocabulary and mapping of vocabularies to indices
        self.build_vocabulary(doc_term_frequencies)

        self.vectorize_documents(doc_term_frequencies)

        print(f"Vocabulary size |V|: {len(self.vocab)} unique terms across {self.N} documents.")
        ## Commented to avoid taking your memory
        # self.save_index()
    def inverted_index(self, doc_term_frequencies):
        ii = defaultdict(dict)
        for doc_id, counts in doc_term_frequencies.items():
            for term, count in counts.items:
                ii[term][doc_id] = count
        return dict(ii)
    def calculate_document_frequencies(self, doc_term_frequencies):
        df_counts = defaultdict(int)
        for counts in doc_term_frequencies.values():
            for term in counts.keys():
                df_counts[term] += 1
        self.df = dict(df_counts)
    def build_vocabulary(self, term_frequencies, max_vocab_size=20000):
        pass
    def vectorize_documents(self, doc_term_frequencies):
        pass
    def save_index(self):
        """Serialize current VSM index state and raw documents to a JSON file."""
        index_data = {
            "N": self.N,
            "vocab": self.vocab,
            "word_to_idx": self.word_to_idx,
            "df": self.df,
            "doc_vectors": self.doc_vectors,
            "documents": self.documents,
        }
        with open(self.index_file_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
        print(f"Index successfully saved to '{self.index_file_path}'.")
    def load_index(self):
        """Load serialized VSM index state and raw documents from a JSON file."""
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file '{self.index_file_path}' does not exist.")

        with open(self.index_file_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.N = index_data["N"]
        self.vocab = index_data["vocab"]
        self.word_to_idx = index_data["word_to_idx"]
        self.df = index_data["df"]
        self.doc_vectors = index_data["doc_vectors"]
        self.documents = index_data["documents"]
        print(f"Index successfully loaded from '{self.index_file_path}'.")
         

    
        
