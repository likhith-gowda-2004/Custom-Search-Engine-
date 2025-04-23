import os
import sys
from flask import Flask, request, render_template, jsonify
import tempfile

# Import our search engine
from search_engine import MiniSearchEngine

app = Flask(__name__)

# Initialize search engine
search_engine = MiniSearchEngine()

# Create sample documents directory if it doesn't exist
SAMPLE_DOCS_DIR = os.path.join(tempfile.gettempdir(), "sample_docs")
os.makedirs(SAMPLE_DOCS_DIR, exist_ok=True)

# Sample documents
SAMPLE_DOCUMENTS = [
    {"title": "Information Retrieval Basics", 
     "content": "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources. Searches can be based on full-text or other content-based indexing. Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data, and for databases of texts, images or sounds."},
    
    {"title": "Search Engine Fundamentals", 
     "content": "A search engine is an information retrieval system designed to help find information stored on a computer system. Search engines help to minimize the time required to find information and the amount of information which must be consulted. The most public visible form of search engines are Web search engines, which search for information on the World Wide Web."},
    
    {"title": "Vector Space Model", 
     "content": "Vector space model or term vector model is an algebraic model for representing text documents as vectors of identifiers, such as index terms. It is used in information filtering, information retrieval, indexing and relevancy rankings. Its first use was in the SMART Information Retrieval System."},
    
    {"title": "TF-IDF Explained", 
     "content": "TF-IDF stands for term frequency–inverse document frequency. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word."},
    
    {"title": "Boolean Retrieval", 
     "content": "Boolean retrieval is a model for information retrieval in which queries are posed in the form of Boolean expressions of terms, that is, terms combined with the operators AND, OR, and NOT. The model views each document as just a set of words."},
    
    {"title": "Inverted Index Structure", 
     "content": "An inverted index is an index data structure storing a mapping from content, such as words or numbers, to its locations in a document or a set of documents. In simple words, it is a hashmap like data structure that directs you from a word to a document or a web page. The inverted index data structure is a central component of a typical search engine indexing algorithm."},
    
    {"title": "Precision and Recall", 
     "content": "In information retrieval, precision is the fraction of retrieved documents that are relevant to the query, while recall is the fraction of relevant documents that are retrieved. Both precision and recall are based on an understanding and measure of relevance. These metrics are used to evaluate search strategies and information retrieval systems."},
    
    {"title": "Query Expansion Techniques", 
     "content": "Query expansion is the process of reformulating a seed query to improve retrieval performance. In the context of search engines, query expansion involves evaluating a user's input and expanding the search query to match additional documents. The goal of query expansion is to produce additional search terms for increasing the likelihood of finding relevant documents."},
    
    {"title": "Web Crawling Basics", 
     "content": "A Web crawler, sometimes called a spider, is an Internet bot that systematically browses the World Wide Web, typically for the purpose of Web indexing. Web search engines and some other sites use Web crawling or spidering software to update their web content or indices of others sites' web content. Web crawlers copy pages for processing by a search engine which indexes the downloaded pages."},
    
    {"title": "Relevance Feedback", 
     "content": "Relevance feedback is a feature of some information retrieval systems. Relevance feedback is a feature of some information retrieval systems. The idea behind relevance feedback is to take the results that are initially returned from a given query, to gather user feedback, and to use information about whether or not those results are relevant to perform a new query."}
]

# Generate sample documents if they don't exist
def generate_sample_documents():
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        file_path = os.path.join(SAMPLE_DOCS_DIR, f"{doc['title'].replace(' ', '_')}.txt")
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc['content'])

# Create HTML templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create HTML template file
with open(os.path.join(TEMPLATES_DIR, "index.html"), 'w', encoding='utf-8') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Mini Search Engine</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .search-container {
            margin-bottom: 20px;
            padding: 20px;
            background-color: #f7f7f7;
            border-radius: 5px;
        }
        .search-box {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .search-button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .search-options {
            margin-top: 10px;
        }
        .result-item {
            margin-bottom: 15px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #fff;
        }
        .result-title {
            font-size: 18px;
            color: #3498db;
            margin-bottom: 5px;
        }
        .result-snippet {
            color: #555;
        }
        .result-score {
            color: #7f8c8d;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Mini Search Engine</h1>
    
    <div class="search-container">
        <input type="text" id="searchBox" class="search-box" placeholder="Enter your search query...">
        <button onclick="performSearch()" class="search-button">Search</button>
        
        <div class="search-options">
            <label><input type="radio" name="searchType" value="tf_idf" checked> TF-IDF</label>
            <label><input type="radio" name="searchType" value="boolean"> Boolean</label>
            <label><input type="radio" name="searchType" value="phrase"> Phrase</label>
            <label><input type="radio" name="searchType" value="bm25"> BM25</label>
            <label><input type="radio" name="searchType" value="zone"> Zone</label>
            <label><input type="radio" name="searchType" value="tiered"> Tiered</label>
            <label><input type="radio" name="searchType" value="expanded"> Query Expansion</label>
        </div>
    </div>
    
    <div id="loading" class="loading">
        <p>Searching...</p>
    </div>
    
    <div id="searchResults">
        <!-- Results will be displayed here -->
    </div>
    
    <script>
        function performSearch() {
            // Get search query and type
            const query = document.getElementById('searchBox').value;
            const searchType = document.querySelector('input[name="searchType"]:checked').value;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('searchResults').innerHTML = '';
            
            // Perform search via AJAX
            fetch(`/search?query=${encodeURIComponent(query)}&type=${searchType}`)
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    document.getElementById('loading').style.display = 'none';
                    
                    // Display results
                    const resultsDiv = document.getElementById('searchResults');
                    
                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = '<p>No results found.</p>';
                        return;
                    }
                    
                    let resultsHtml = `<h2>Search Results (${data.results.length})</h2>`;
                    
                    data.results.forEach(result => {
                        resultsHtml += `
                            <div class="result-item">
                                <div class="result-title">${result.title}</div>
                                <div class="result-snippet">${result.snippet}</div>
                                <div class="result-score">Score: ${result.score.toFixed(4)}</div>
                            </div>
                        `;
                    });
                    
                    resultsDiv.innerHTML = resultsHtml;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('searchResults').innerHTML = 
                        '<p>An error occurred while searching. Please try again.</p>';
                });
        }
        
        // Allow pressing Enter key to search
        document.getElementById('searchBox').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                performSearch();
            }
        });
    </script>
</body>
</html>
    """)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    search_type = request.args.get('type', 'tf_idf')
    
    results = search_engine.get_search_results(query, search_type)
    
    return jsonify({
        'query': query,
        'type': search_type,
        'results': results
    })

@app.route('/rebuild_index')
def rebuild_index():
    search_engine.build_index_from_directory(SAMPLE_DOCS_DIR)
    return jsonify({'status': 'success', 'message': 'Index rebuilt successfully'})

if __name__ == '__main__':
    # Generate sample documents
    generate_sample_documents()
    
    # Build index
    search_engine.build_index_from_directory(SAMPLE_DOCS_DIR)
    
    # Start the Flask app
    app.run(debug=True)