"""preprocessing and TF/IDF"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

#tokenization method
def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    tokens = re.findall(r'\b[a-z]+\b', clean_text.lower())
    return [token for token in tokens if token not in STOPWORDS]

class tfidf:
    def __init__(self, index_file_path= "Study.json"):
        #file path
        self.index_file_path = index_file_path
        self.vocab = []
        self.doc_vectors = {}
        self.word_to_idx = {}
        self.documents = {}
        #document frequencies
        self.df = {}
        # number of documents
        self.N = 0
        #length of documents
        self.doc_lengths = {}
        #average length of documents
        self.doc_avg_lengths = 0
        #the inverted index
        self.invert_index = {}
    #indexing for papers 
    def index_file(self, data_path, force_reindex=False):
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
            doc_id = doc["paper_id"]
            # For results representation
            self.documents[doc_id] = doc
            text = doc.get("title", "") + " " + doc.get("abstract", "")
            tokens = tokenize(text)
            # get term/token frequency (just in the current doc/answer)
            counts = Counter(tokens)
            doc_term_frequencies[doc_id] = counts

        # Calculates document frequency for each term
        self.calculate_document_frequencies(doc_term_frequencies)
        self.calculate_doc_length(doc_term_frequencies)
        # Build vocabulary and mapping of vocabularies to indices
        #self.build_vocabulary(doc_term_frequencies)
        self.inverted_index(doc_term_frequencies)
        #self.vectorize_documents(doc_term_frequencies)

        self.save_index()

        print(f"indexed {self.N} papers")

    #indexing for paragraphs
    def index_file_paragraph(self, data_path, force_reindex=False):
        if os.path.exists(self.index_file_path) and not force_reindex:
            print(f"Index file '{self.index_file_path}' found. Loading from disk...")
            self.load_index()
            return

        print(f"Indexing source file '{data_path}'...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        doc_term_frequencies = {}

        for paper in data:
            for paragraph_id, text in paper ["paragraphs"].items():
                paragraph_id = str(paragraph_id)
                tokens = tokenize(text)
                doc_term_frequencies[paragraph_id] = Counter(tokens)
        self.N = len(doc_term_frequencies)
        self.calculate_document_frequencies(doc_term_frequencies)
        self.calculate_doc_length(doc_term_frequencies)
        self.inverted_index(doc_term_frequencies)

        self.save_index()
        print(f"indexed {self.N} papers")
    
    #creates inverted index
    def inverted_index(self, doc_term_frequencies):
        ii = defaultdict(dict)
        for doc_id, counts in doc_term_frequencies.items():
            for term, count in counts.items():
                ii[term][doc_id] = count
        self.invert_index = dict(ii)
        return self.invert_index
    
    def vectorize_documents(self, doc_term_frequencies):
        vocab_size = len(self.vocab)
        self.doc_vectors = {}
        for doc_id, counts in doc_term_frequencies.items():
            doc_vector = [0.0] * vocab_size
            for term, count in counts.items():
                if term in self.word_to_idx:
                    pos = self.word_to_idx[term]
                    doc_vector[pos] = self.calculate_tf_idf(count, self.df[term])
            self.doc_vectors[doc_id] = doc_vector
    #calculates document frequencies
    def calculate_document_frequencies(self, doc_term_frequencies):
        df_counts = defaultdict(int)
        for counts in doc_term_frequencies.values():
            for term in counts.keys():
                df_counts[term] += 1
        self.df = dict(df_counts)

    def build_vocabulary(self, term_frequencies, max_vocab_size=20000):
        collection_frequencies = Counter()
        for counts in term_frequencies.values():
            collection_frequencies.update(counts)
        if max_vocab_size and max_vocab_size < len(collection_frequencies):
            kterms = [term for term, _ in collection_frequencies.most_common(max_vocab_size)]
        else:
            kterms = list(collection_frequencies.keys())
        self.vocab = sorted(list(kterms))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
    #calculates tfidf
    def calculate_tf_idf(self, count, df_val):
        if count <= 0 or df_val <= 0 or self.N == 0:
            return 0
        tf = 1 + math.log10(count)
        idf = math.log10(self.N / df_val)
        return tf * idf
    
    #calculates doc lengths
    def calculate_doc_length(self, doc_term_frequencies):
        self.doc_lengths = {
            doc_id: sum(counts.values())
            for doc_id, counts in doc_term_frequencies.items()
        }
        if self.N > 0:
            self.doc_avg_length =(
                sum(self.doc_lengths.values()) / self.N
            )
        else:
            self.doc_avg_length = 0.0

    
    def save_index(self):
        index_data = {
            "inverted_index": self.invert_index,
            #"vocab": self.vocab,
            #"word_to_idx": self.word_to_idx,
            "df": self.df,
            #"doc_vectors": self.doc_vectors,
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "num_docs": self.N,
            "avg_doc_length": self.doc_avg_length,
        }
        with open(self.index_file_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
        
    def load_index(self):
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file '{self.index_file_path}' does not exist.")

        with open(self.index_file_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.invert_index = index_data["inverted_index"]
        #self.vocab = index_data["vocab"]
        #self.word_to_idx = index_data["word_to_idx"]
        self.df = index_data["df"]
        #self.doc_vectors = index_data["doc_vectors"]
        #self.documents = index_data["documents"]
        self.doc_lengths = index_data["doc_lengths"]
        self.N = index_data["num_docs"]
        self.doc_avg_length = index_data["avg_doc_length"]
        print(f"Index successfully loaded from '{self.index_file_path}'.")
    def main():
        pass
             

    
        
