import nltk
import math
import numpy

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')


# NLTK integrated Inl2 retrieval functionality (MP Integration Function #1)
def get_stopwords(lang='english'):
    return set(nltk.corpus.stopwords.words(lang))


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


def get_text_sentiment(text, negative_thres=-0.05, positive_thres=0.05):
    # Create an instance of SentimentIntensityAnalyzer
    from nltk.sentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment scores
    sentiment_score = sid.polarity_scores(text)

    # Determine sentiment based on the compound score
    if sentiment_score['compound'] >= positive_thres:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= negative_thres:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_score


def max_entropy(train_data, test_data, algorithm='GIS', trace=0, max_iter=1000):
    # Feature extraction function
    def document_features(document):
        return {word: (word in document) for word in document_words}

    # Get all unique words in the dataset
    document_words = set(word.lower() for doc, _ in train_data for word in doc)

    # Train the MaxEnt classifier
    classifier = nltk.classify.MaxentClassifier.train(train_data, algorithm=algorithm, trace=trace, max_iter=max_iter)

    # Evaluate the classifier
    probs = []
    for featureset in test_data:
        pdist = classifier.prob_classify(featureset)
        prob = {}
        for word in ['x', 'y']:
            prob[word] = round(pdist.prob(word), 2)
        probs.append(prob)

    # Get prediction for test data
    many = classifier.classify_many(test_data)

    return classifier, probs, many


def analyze_collocation(text, gram):
    # Tools to utilize depending on the gram number
    finders = {
        2: nltk.collocations.BigramCollocationFinder,
        3: nltk.collocations.TrigramCollocationFinder,
        4: nltk.collocations.QuadgramCollocationFinder
    }
    measures = {
        2: nltk.collocations.BigramAssocMeasures,
        3: nltk.collocations.TrigramAssocMeasures,
        4: nltk.collocations.QuadgramAssocMeasures
    }

    # Tokenize the text
    words = nltk.wordpunct_tokenize(text)

    # Create a gramCollocationFinder
    gram_finder = finders[gram].from_words(words)

    # Filter out collocations based on frequency and other measures
    gram_measures = measures[gram]()

    return {'gram_finder': gram_finder, 'gram_measures': gram_measures}


def get_collocation_ngram_score(text, gram=2, n=2, method='raw_freq'):
    analysis = analyze_collocation(text, gram)
    return analysis['gram_finder'].score_ngrams(getattr(analysis['gram_measures'], method))


def get_collocation_n_best(text, gram=2, n=2, method='raw_freq'):
    analysis = analyze_collocation(text, gram)
    return analysis['gram_finder'].nbest(getattr(analysis['gram_measures'], method), n)


def analyze_corpus(corpus):
    # Tokenize and remove stopwords
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in corpus]
    stop_words = get_stopwords()
    filtered_corpus = [[word.lower() for word in doc if word.isalnum() and word not in stop_words] for doc in tokenized_corpus]

    # Perform part-of-speech tagging
    pos_tagged_corpus = [nltk.pos_tag(doc) for doc in filtered_corpus]

    return pos_tagged_corpus


def generate_tree(text, grammar, text_is_formatted=False):
    def build_trees_from_text():
        # Tokenize the input string
        words = nltk.word_tokenize(text)

        # Perform syntactic parsing
        parser = nltk.ChartParser(grammar)
        return list(parser.parse(words))

    if text_is_formatted:
        trees = list(nltk.tree.Tree.fromstring(text))
    else:
        trees = build_trees_from_text()

    for tree in trees:
        # Visualize the tree
        tree.pretty_print()

    return trees
