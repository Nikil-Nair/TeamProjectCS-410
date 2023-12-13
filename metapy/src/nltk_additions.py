import nltk
import math
import numpy

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# NLTK integrated Inl2 retrieval functionality (MP Integration Function #1)
def inl2_retrieval(documents, query):    
    # Tokenize documents and query
    def tokenize(text):
        return nltk.word_tokenize(text.lower())

    t_doc = [tokenize(doc) for doc in documents]
    t_query = tokenize(query)

    # Compute TF-IDF scores for each document
    def score(query_terms, document_terms, corpus):
        tf_scores = [document_terms.count(term) for term in query_terms]
        idf_scores = [math.log(len(corpus) / (1 + corpus.count(term))) for term in query_terms]
        tfidf_scores = [tf * idf for tf, idf in zip(tf_scores, idf_scores)]
        return sum(tfidf_scores)

    corpus = [token for doc in t_doc for token in doc]
    scores = [(doc, score(t_query, doc, corpus)) for doc in t_doc]

    # Sort documents by score (highest score first)
    scores.sort(key=lambda x: x[1], reverse=True)
        
    return scores

# NLTK integrated Part of speech tagging functionality (MP Integration Function #2)
def pos_tagging(text):
    # Tokenize the input text into words
    words = nltk.word_tokenize(text)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)

    return pos_tags

# NDCG scoring functionality (MP Integration Function #3)
def ndcg(ranker, queries, relevant_docs, k=10):
    ndcg_scores = []

    def dcg(relevance_scores):
        # Calculate the Discounted Cumulative Gain (DCG)
        dcg_score = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg_score += (2**rel - 1) / math.log2(i + 2)
        return dcg_score

    for query in queries:
        # Use the ranker to rank documents for the current query
        ranked_docs = ranker(query)[:k]  # Consider only the top k documents

        # Calculate the relevance scores for the ranked documents
        relevance_scores = [1 if doc in relevant_docs.get(query, []) else 0 for doc in ranked_docs]

        # Calculate DCG at k
        dcg_at_k = dcg(relevance_scores)

        # Sort the relevance scores in descending order for ideal DCG calculation
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)

        # Calculate ideal DCG at k
        ideal_dcg_at_k = dcg(ideal_relevance_scores)

        # Calculate NDCG at k
        if ideal_dcg_at_k == 0:
            ndcg_score = 0.0
        else:
            ndcg_score = dcg_at_k / ideal_dcg_at_k

        ndcg_scores.append(ndcg_score)

    # Calculate the average NDCG over all queries
    avg_ndcg = numpy.mean(ndcg_scores)

    return avg_ndcg

# NLTK integrated Naive Bayes Classifier (MP Integration Function #4)
def naive_bayes_classifier(training_data, new_text):
    # Define a feature extractor function (simple bag-of-words)
    def extract_features(text):
        words = set(text)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in words)
        return features

    # Get the most frequent words as features
    all_words = nltk.FreqDist(w.lower() for w in nltk.word_tokenize(' '.join([review for review, _ in training_data])))
    word_features = list(all_words.keys())[:2000]  # Use the top 2000 words as features

    # Extract features for each review
    feature_sets = [(extract_features(review.split()), label) for (review, label) in training_data]

    # Train a Naive Bayes classifier
    classifier = nltk.classify.NaiveBayesClassifier.train(feature_sets)

    # Classify the new text
    new_features = extract_features(new_text.split())
    classification = classifier.classify(new_features)

    return classification

# NLTK integrated stemmer and lemmatizer (MP Integration Function #5)
def stem_lemmatize(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Initialize the stemmer and lemmatizer
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Apply stemming and lemmatization to each word
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return stemmed_words, lemmatized_words
