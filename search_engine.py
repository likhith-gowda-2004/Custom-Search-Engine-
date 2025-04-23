import os
import re
import math
import json
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class Document:
    def __init__(self, doc_id, content, title=None):
        self.id = doc_id
        self.content = content
        self.title = title or f"Document {doc_id}"
        self.length = len(self.content.split())  # Used for normalization

class MiniSearchEngine:
    def __init__(self):
        self.documents = {}
        self.inverted_index = defaultdict(list)
        self.positional_index = defaultdict(lambda: defaultdict(list))
        self.document_vectors = {}
        self.idf_scores = {}
        self.avg_doc_length = 0
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # For zone indexing
        self.title_index = defaultdict(list)
        
        # For tiered indexing
        self.tier1_index = defaultdict(list)  # Common terms
        self.tier2_index = defaultdict(list)  # Medium frequency
        self.tier3_index = defaultdict(list)  # Rare terms

    def preprocess_text(self, text):
        """Tokenize, remove stopwords, and stem the text."""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and stem
        processed_tokens = [self.stemmer.stem(token) for token in tokens 
                          if token.isalnum() and token not in self.stop_words]
        
        return processed_tokens

    def add_document(self, doc_id, content, title=None):
        """Add a document to the search engine."""
        document = Document(doc_id, content, title)
        self.documents[doc_id] = document
        
        # Process the content
        tokens = self.preprocess_text(content)
        positions = {}
        
        # Build positions dictionary
        for position, token in enumerate(tokens):
            if token not in positions:
                positions[token] = []
            positions[token].append(position)
        
        # Update inverted index
        for token, pos_list in positions.items():
            self.inverted_index[token].append((doc_id, len(pos_list)))
            self.positional_index[token][doc_id] = pos_list
        
        # Process title for zone indexing
        if title:
            title_tokens = self.preprocess_text(title)
            for token in set(title_tokens):
                self.title_index[token].append(doc_id)

    def build_index_from_directory(self, directory_path):
        """Build an index from text files in a directory."""
        doc_id = 0
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.add_document(doc_id, content, title=filename)
                    doc_id += 1
        
        # Calculate average document length for BM25
        total_length = sum(doc.length for doc in self.documents.values())
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        
        # Calculate IDF scores
        self._calculate_idf_scores()
        
        # Build document vectors and tiered index
        self._build_document_vectors()
        self._build_tiered_index()

    def _calculate_idf_scores(self):
        """Calculate IDF scores for all terms in the corpus."""
        N = len(self.documents)
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf_scores[term] = math.log(N / df) if df > 0 else 0

    def _build_document_vectors(self):
        """Build TF-IDF vectors for all documents."""
        for doc_id, document in self.documents.items():
            vector = {}
            # Calculate term frequencies in this document
            tokens = self.preprocess_text(document.content)
            term_frequencies = Counter(tokens)
            
            # Calculate TF-IDF for each term
            for term, tf in term_frequencies.items():
                if term in self.idf_scores:
                    # TF-IDF weighting
                    vector[term] = tf * self.idf_scores[term]
            
            # Store the document vector
            self.document_vectors[doc_id] = vector

    def _build_tiered_index(self):
        """Build a tiered index based on term frequency."""
        # Calculate global term frequencies
        term_freqs = {}
        for term, postings in self.inverted_index.items():
            term_freqs[term] = sum(freq for _, freq in postings)
        
        # Sort terms by frequency
        sorted_terms = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # Divide into tiers
        n_terms = len(sorted_terms)
        tier1_cutoff = n_terms // 3
        tier2_cutoff = 2 * n_terms // 3
        
        # Assign terms to tiers
        for i, (term, _) in enumerate(sorted_terms):
            if i < tier1_cutoff:
                self.tier1_index[term] = self.inverted_index[term]
            elif i < tier2_cutoff:
                self.tier2_index[term] = self.inverted_index[term]
            else:
                self.tier3_index[term] = self.inverted_index[term]

    def boolean_search(self, query):
        """Perform a boolean search (AND, OR, NOT)."""
        # Simple parsing of boolean query
        tokens = query.split()
        i = 0
        result_set = set()
        current_op = "OR"  # Default operator
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.upper() in ["AND", "OR", "NOT"]:
                current_op = token.upper()
                i += 1
                continue
            
            # Process the term
            term = self.stemmer.stem(token.lower())
            docs_with_term = {doc_id for doc_id, _ in self.inverted_index.get(term, [])}
            
            # Apply the current operator
            if current_op == "AND":
                if not result_set:  # First term
                    result_set = docs_with_term
                else:
                    result_set &= docs_with_term
            elif current_op == "OR":
                result_set |= docs_with_term
            elif current_op == "NOT":
                result_set -= docs_with_term
            
            i += 1
        
        return sorted(result_set)

    def phrase_search(self, phrase):
        """Search for an exact phrase."""
        # Preprocess the phrase
        phrase_tokens = self.preprocess_text(phrase)
        
        if not phrase_tokens:
            return []
        
        # Get documents containing the first term
        first_term = phrase_tokens[0]
        candidate_docs = set(self.positional_index[first_term].keys())
        
        # Filter for documents containing all terms
        for token in phrase_tokens[1:]:
            docs_with_token = set(self.positional_index[token].keys())
            candidate_docs &= docs_with_token
        
        # Check for consecutive positions
        results = []
        for doc_id in candidate_docs:
            positions = self.positional_index[phrase_tokens[0]][doc_id]
            
            for pos in positions:
                match = True
                for i, token in enumerate(phrase_tokens[1:], 1):
                    if pos + i not in self.positional_index[token][doc_id]:
                        match = False
                        break
                
                if match:
                    results.append(doc_id)
                    break  # Found a match in this document
        
        return sorted(results)

    def proximity_search(self, term1, term2, max_distance):
        """Find documents where term1 and term2 are within max_distance of each other."""
        term1 = self.stemmer.stem(term1.lower())
        term2 = self.stemmer.stem(term2.lower())
        
        # Get documents containing both terms
        docs_with_term1 = set(self.positional_index[term1].keys())
        docs_with_term2 = set(self.positional_index[term2].keys())
        candidate_docs = docs_with_term1.intersection(docs_with_term2)
        
        results = []
        for doc_id in candidate_docs:
            positions1 = self.positional_index[term1][doc_id]
            positions2 = self.positional_index[term2][doc_id]
            
            for pos1 in positions1:
                for pos2 in positions2:
                    if abs(pos1 - pos2) <= max_distance:
                        results.append(doc_id)
                        break
                else:
                    continue
                break
        
        return sorted(results)

    def tf_idf_search(self, query):
        """Search using TF-IDF and Vector Space Model."""
        # Create a query vector
        query_tokens = self.preprocess_text(query)
        query_tf = Counter(query_tokens)
        query_vector = {}
        
        for term, tf in query_tf.items():
            if term in self.idf_scores:
                query_vector[term] = tf * self.idf_scores[term]
        
        # Calculate cosine similarity with each document
        scores = {}
        for doc_id, doc_vector in self.document_vectors.items():
            # Skip empty documents
            if not doc_vector:
                continue
            
            # Calculate dot product
            dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) 
                              for term in set(query_vector) | set(doc_vector))
            
            # Calculate magnitudes
            query_magnitude = math.sqrt(sum(score**2 for score in query_vector.values()))
            doc_magnitude = math.sqrt(sum(score**2 for score in doc_vector.values()))
            
            # Calculate cosine similarity
            if query_magnitude > 0 and doc_magnitude > 0:
                scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
            else:
                scores[doc_id] = 0
        
        # Sort by score
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results

    def bm25_search(self, query, k1=1.5, b=0.75):
        """Search using BM25 ranking algorithm."""
        query_tokens = self.preprocess_text(query)
        scores = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
                
            idf = self.idf_scores[term]
            
            for doc_id, freq in self.inverted_index[term]:
                doc_length = self.documents[doc_id].length
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort by score
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results

    def zone_search(self, query, title_weight=2.0, content_weight=1.0):
        """Search with zone indexing (title and content)."""
        query_tokens = self.preprocess_text(query)
        scores = defaultdict(float)
        
        for term in query_tokens:
            # Score based on title
            for doc_id in self.title_index.get(term, []):
                scores[doc_id] += title_weight
            
            # Score based on content (TF-IDF)
            if term in self.inverted_index:
                for doc_id, freq in self.inverted_index[term]:
                    scores[doc_id] += content_weight * freq * self.idf_scores[term]
        
        # Sort by score
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results

    def tiered_search(self, query):
        """Search using tiered indexing for better performance."""
        query_tokens = set(self.preprocess_text(query))
        
        # Check which tier to search in
        tier1_terms = query_tokens.intersection(self.tier1_index.keys())
        tier2_terms = query_tokens.intersection(self.tier2_index.keys())
        tier3_terms = query_tokens.intersection(self.tier3_index.keys())
        
        # Start with the rarest terms (tier3) for better performance
        candidate_docs = set()
        if tier3_terms:
            for term in tier3_terms:
                candidate_docs.update(doc_id for doc_id, _ in self.tier3_index[term])
        elif tier2_terms:
            for term in tier2_terms:
                candidate_docs.update(doc_id for doc_id, _ in self.tier2_index[term])
        elif tier1_terms:
            for term in tier1_terms:
                candidate_docs.update(doc_id for doc_id, _ in self.tier1_index[term])
        
        # Now score the candidates using tf-idf
        scores = {}
        query_vector = {term: 1 for term in query_tokens}  # Simple binary vector
        
        for doc_id in candidate_docs:
            doc_vector = self.document_vectors[doc_id]
            
            # Calculate dot product
            dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) 
                              for term in query_tokens)
            
            # Calculate magnitudes
            query_magnitude = math.sqrt(len(query_tokens))
            doc_magnitude = math.sqrt(sum(score**2 for score in doc_vector.values()))
            
            # Calculate cosine similarity
            if doc_magnitude > 0:
                scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
        
        # Sort by score
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results

    def query_expansion(self, query, num_expansions=2):
        """Expand the query with related terms from top documents."""
        # First get top documents using standard TF-IDF search
        initial_results = self.tf_idf_search(query)[:3]  # Get top 3 documents
        
        # Extract potential expansion terms
        expansion_candidates = Counter()
        for doc_id, _ in initial_results:
            content = self.documents[doc_id].content
            tokens = self.preprocess_text(content)
            for token in tokens:
                expansion_candidates[token] += 1
        
        # Remove original query terms
        query_tokens = set(self.preprocess_text(query))
        for token in query_tokens:
            if token in expansion_candidates:
                del expansion_candidates[token]
        
        # Get top expansion terms
        expansion_terms = [term for term, _ in expansion_candidates.most_common(num_expansions)]
        
        # Build expanded query
        expanded_query = query + " " + " ".join(expansion_terms)
        
        # Search with expanded query
        return self.tf_idf_search(expanded_query)

    def save_index(self, filename):
        """Save the index to a file."""
        index_data = {
            'documents': {str(k): {'title': v.title, 'length': v.length} 
                         for k, v in self.documents.items()},
            'inverted_index': {k: v for k, v in self.inverted_index.items()},
            'positional_index': {k: {str(dk): dv for dk, dv in v.items()} 
                               for k, v in self.positional_index.items()},
            'idf_scores': self.idf_scores,
            'avg_doc_length': self.avg_doc_length,
            'title_index': {k: v for k, v in self.title_index.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(index_data, f)

    def load_index(self, filename):
        """Load the index from a file."""
        with open(filename, 'r') as f:
            index_data = json.load(f)
        
        # Reconstruct the documents
        self.documents = {}
        for doc_id_str, doc_data in index_data['documents'].items():
            doc_id = int(doc_id_str)
            doc = Document(doc_id, "", doc_data['title'])
            doc.length = doc_data['length']
            self.documents[doc_id] = doc
        
        # Load the indexes
        self.inverted_index = defaultdict(list, index_data['inverted_index'])
        
        # Load positional index
        self.positional_index = defaultdict(lambda: defaultdict(list))
        for term, postings in index_data['positional_index'].items():
            for doc_id_str, positions in postings.items():
                self.positional_index[term][int(doc_id_str)] = positions
        
        # Load other data
        self.idf_scores = index_data['idf_scores']
        self.avg_doc_length = index_data['avg_doc_length']
        self.title_index = defaultdict(list, index_data['title_index'])
        
        # Rebuild document vectors
        self._build_document_vectors()
        
        # Rebuild tiered index
        self._build_tiered_index()

    def get_document_snippet(self, doc_id, query, context_size=40):
        """Get a snippet of the document that contains the query terms."""
        if doc_id not in self.documents:
            return ""
        
        content = self.documents[doc_id].content
        query_terms = self.preprocess_text(query)
        
        # Find positions of query terms in the content
        content_tokens = content.split()
        content_lower = [token.lower() for token in content_tokens]
        
        best_position = -1
        max_matches = 0
        
        # Find the best position with the most query term matches
        for i in range(len(content_tokens)):
            matches = 0
            for term in query_terms:
                # Check if term appears in a window of size context_size around position i
                window = ' '.join(content_tokens[max(0, i-context_size//2):min(len(content_tokens), i+context_size//2)])
                if term in self.stemmer.stem(window.lower()):
                    matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_position = i
        
        # Get snippet around the best position
        start = max(0, best_position - context_size//2)
        end = min(len(content_tokens), best_position + context_size//2)
        
        snippet = ' '.join(content_tokens[start:end])
        if start > 0:
            snippet = "..." + snippet
        if end < len(content_tokens):
            snippet = snippet + "..."
            
        return snippet

    def get_search_results(self, query, search_type='tf_idf', limit=10):
        """Get formatted search results."""
        if search_type == 'boolean':
            doc_ids = self.boolean_search(query)
            results = [(doc_id, 1.0) for doc_id in doc_ids]  # All have same score in boolean search
        elif search_type == 'phrase':
            doc_ids = self.phrase_search(query)
            results = [(doc_id, 1.0) for doc_id in doc_ids]
        elif search_type == 'tf_idf':
            results = self.tf_idf_search(query)
        elif search_type == 'bm25':
            results = self.bm25_search(query)
        elif search_type == 'zone':
            results = self.zone_search(query)
        elif search_type == 'tiered':
            results = self.tiered_search(query)
        elif search_type == 'expanded':
            results = self.query_expansion(query)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Format the results
        formatted_results = []
        for doc_id, score in results[:limit]:
            doc = self.documents[doc_id]
            snippet = self.get_document_snippet(doc_id, query)
            formatted_results.append({
                'doc_id': doc_id,
                'title': doc.title,
                'score': score,
                'snippet': snippet
            })
        
        return formatted_results


# Test the search engine
if __name__ == "__main__":
    # Create a search engine
    search_engine = MiniSearchEngine()
    
    # Add some sample documents
    search_engine.add_document(0, "Information retrieval is the science of searching for information in documents.", 
                              "Information Retrieval")
    search_engine.add_document(1, "The science of information retrieval has many applications.", 
                              "IR Applications")
    search_engine.add_document(2, "Search engines are the most visible applications of information retrieval.", 
                              "Search Engines")
    search_engine.add_document(3, "Efficient information retrieval is challenging due to the size of the web.", 
                              "IR Challenges")
    search_engine.add_document(4, "The goal of information retrieval is to find relevant information quickly.", 
                              "IR Goals")
    
    # Example searches
    print("Boolean search for 'information AND retrieval':")
    results = search_engine.boolean_search("information AND retrieval")
    print([search_engine.documents[doc_id].title for doc_id in results])
    
    print("\nPhrase search for 'information retrieval':")
    results = search_engine.phrase_search("information retrieval")
    print([search_engine.documents[doc_id].title for doc_id in results])
    
    print("\nTF-IDF search for 'information retrieval':")
    results = search_engine.tf_idf_search("information retrieval")
    for doc_id, score in results[:3]:
        print(f"{search_engine.documents[doc_id].title}: {score:.4f}")