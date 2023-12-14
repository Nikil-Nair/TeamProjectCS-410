import unittest
import nltk
import math
import numpy

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Import the functions to be tested
from nltk_additions import *


class TestYourFunctions(unittest.TestCase):

    # Test NLTK integrated inl2 retrieval function
    def test_inl2_retrieval(self):
        documents = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one."
        ]
        query = "first document"
        result = inl2_retrieval(documents, query)

        eq = [(['this', 'is', 'the', 'first', 'document', '.'], 3.9120230054281464),
              (['this', 'document', 'is', 'the', 'second', 'document', '.'], 3.2188758248682006),
              (['and', 'this', 'is', 'the', 'third', 'one', '.'], 0.0)]
        self.assertEqual(result, eq)

        documents = [
            "This is the second test document.",
            "Could this be another sample document?",
            "Let's run some more tests on this document?"
        ]
        query = "a second document that will be sampled for tests"
        result = inl2_retrieval(documents, query)

        eq = [(['this', 'is', 'the', 'second', 'test', 'document', '.'], 4.276666119016055),
              (['could', 'this', 'be', 'another', 'sample', 'document', '?'], 4.276666119016055),
              (['let', "'s", 'run', 'some', 'more', 'tests', 'on', 'this', 'document', '?'], 4.276666119016055)]
        self.assertEqual(result, eq)

        documents = [
            "This is the third test we will perform.",
            "I wonder what range of values we will get.",
            "Let's throw in our names: [Nikil, David]."
        ]
        query = "a third document that Nikil uses as a range of values"
        result = inl2_retrieval(documents, query)

        eq = [(['i', 'wonder', 'what', 'range', 'of', 'values', 'we', 'will', 'get', '.'], 8.317766166719343),
              (['this', 'is', 'the', 'third', 'test', 'we', 'will', 'perform', '.'], 2.772588722239781), (
                  ['let', "'s", 'throw', 'in', 'our', 'names', ':', '[', 'nikil', ',', 'david', ']', '.'],
                  2.772588722239781)]
        self.assertEqual(result, eq)

    # Test NLTK integrated part of speech tagging function
    def test_pos_tagging(self):
        text = "NLTK is a powerful library for natural language processing."
        tags = pos_tagging(text)

        eq = [('NLTK', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('powerful', 'JJ'), ('library', 'NN'), ('for', 'IN'),
              ('natural', 'JJ'), ('language', 'NN'), ('processing', 'NN'), ('.', '.')]
        self.assertEqual(tags, eq)

        text = "Another test to ensure this function works."
        tags = pos_tagging(text)

        eq = [('Another', 'DT'), ('test', 'NN'), ('to', 'TO'), ('ensure', 'VB'), ('this', 'DT'), ('function', 'NN'),
              ('works', 'VBZ'), ('.', '.')]
        self.assertEqual(tags, eq)

        text = "A final test to verify correctness."
        tags = pos_tagging(text)

        eq = [('A', 'DT'), ('final', 'JJ'), ('test', 'NN'), ('to', 'TO'), ('verify', 'VB'), ('correctness', 'NN'),
              ('.', '.')]
        self.assertEqual(tags, eq)

    # Test with a perfect ranking where all relevant documents are at the top
    def test_ndcg_perfect_ranking(self):
        ranker = lambda query: ["doc1", "doc2", "doc3", "doc4"]
        queries = ["query1"]
        relevant_docs = {
            "query1": ["doc1", "doc2", "doc3"]
        }
        k = 3
        result = ndcg(ranker, queries, relevant_docs, k)
        self.assertAlmostEqual(result, 1.0, places=2, msg="Perfect ranking should result in NDCG=1.0")

    # Test with a partial ranking where only some relevant documents are at the top
    def test_ndcg_partial_ranking(self):
        ranker = lambda query: ["doc1", "doc4", "doc2", "doc3"]
        queries = ["query1"]
        relevant_docs = {
            "query1": ["doc1", "doc2", "doc3"]
        }
        k = 3
        result = ndcg(ranker, queries, relevant_docs, k)
        self.assertAlmostEqual(result, 0.919, places=2, msg="Partial ranking should result in NDCG ~0.794")

    # Test with no relevant documents in the ranking
    def test_ndcg_no_relevant_docs(self):
        ranker = lambda query: ["doc4", "doc5", "doc6"]
        queries = ["query1"]
        relevant_docs = {
            "query1": ["doc1", "doc2", "doc3"]
        }
        k = 3
        result = ndcg(ranker, queries, relevant_docs, k)
        self.assertAlmostEqual(result, 0.0, places=2, msg="No relevant docs in ranking should result in NDCG=0.0")

    # Test the classifier with positive training data
    def test_nbc_positive(self):
        training_data = [
            ("This is a positive review", "positive"),
            ("Great product, highly recommended", "positive"),
            ("I love this!", "positive")
        ]
        new_text = "This product is amazing"
        result = naive_bayes_classifier(training_data, new_text)
        self.assertEqual(result, "positive", "Classification should be positive")

    # Test the classifier with negative training data
    def test_nbc_negative(self):
        training_data = [
            ("Terrible experience, do not buy", "negative"),
            ("Waste of money", "negative"),
            ("I regret buying this", "negative")
        ]
        new_text = "I'm so disappointed with this product"
        result = naive_bayes_classifier(training_data, new_text)
        self.assertEqual(result, "negative", "Classification should be negative")

    # Test the classifier with neutral training data
    def test_nbc_neutral(self):
        training_data = [
            ("It's okay, not great but not terrible", "neutral"),
            ("Average product, nothing special", "neutral"),
            ("I have mixed feelings about this", "neutral")
        ]
        new_text = "It's neither good nor bad"
        result = naive_bayes_classifier(training_data, new_text)
        self.assertEqual(result, "neutral", "Classification should be neutral")

    # Test the function's stemming capability
    def test_stemming(self):
        text = "running jumps"
        stemmed_words, _ = stem_lemmatize(text)
        self.assertEqual(stemmed_words, ["run", "jump"], "Stemming should produce ['run', 'jump']")

    # Test the function's lemmatization capability
    def test_lemmatization(self):
        text = "better best"
        _, lemmatized_words = stem_lemmatize(text)
        self.assertEqual(lemmatized_words, ["better", "best"], "Lemmatization should produce ['better', 'best']")

    # Test both stemming and lemmatization together
    def test_stem_lemmatize_combined(self):
        text = "running better"
        stemmed_words, lemmatized_words = stem_lemmatize(text)
        self.assertEqual(stemmed_words, ["run", "better"], "Stemming should produce ['run', 'better']")
        self.assertEqual(lemmatized_words, ["running", "better"], "Lemmatization should produce ['running', 'better']")

    # Test a declarative statement for neutral sentimentality
    def test_neutral_sentiment(self, negative_thres=-0.05, positive_thres=0.05):
        text = 'This is a test.'
        sentiment, score = get_text_sentiment(text, negative_thres, positive_thres)
        self.assertGreaterEqual(score['compound'], negative_thres)
        self.assertLessEqual(score['compound'], positive_thres)

    # Test a negative statement for negative sentimentality
    def test_negative_sentiment(self, negative_thres=-0.05, positive_thres=0.05):
        text = 'This failing test will fail.'
        sentiment, score = get_text_sentiment(text, negative_thres, positive_thres)
        self.assertLessEqual(score['compound'], negative_thres)

    # Test a positive statement for positive sentimentality
    def test_positive_sentiment(self, negative_thres=-0.05, positive_thres=0.05):
        text = 'This successful test will pass.'
        sentiment, score = get_text_sentiment(text, negative_thres, positive_thres)
        self.assertGreaterEqual(score['compound'], positive_thres)

    def test_max_entropy(self):
        train = [
            (dict(a=1, b=1, c=1), 'y'),
            (dict(a=1, b=1, c=1), 'x'),
            (dict(a=1, b=1, c=0), 'y'),
            (dict(a=0, b=1, c=1), 'x'),
            (dict(a=0, b=1, c=1), 'y'),
            (dict(a=0, b=0, c=1), 'y'),
            (dict(a=0, b=1, c=0), 'x'),
            (dict(a=0, b=0, c=0), 'x'),
            (dict(a=0, b=1, c=1), 'y'),
            (dict(a=None, b=1, c=0), 'x'),
        ]
        test = [
            (dict(a=1, b=0, c=1)),  # unseen
            (dict(a=1, b=0, c=0)),  # unseen
            (dict(a=0, b=1, c=1)),  # seen 3 times, labels=y,y,x
            (dict(a=0, b=1, c=0)),  # seen 1 time, label=x
        ]

        gis = max_entropy(train_data=train, test_data=test, algorithm='GIS')
        self.assertEqual([
            {'x': 0.16, "y": 0.84},
            {'x': 0.46, "y": 0.54},
            {'x': 0.41, "y": 0.59},
            {'x': 0.76, "y": 0.24}
        ], gis[1])
        self.assertEqual(['y', 'y', 'y', 'x'], gis[2])

        iis = max_entropy(train_data=train, test_data=test, algorithm='IIS')
        self.assertEqual([
            {'x': 0.16, "y": 0.84},
            {'x': 0.46, "y": 0.54},
            {'x': 0.41, "y": 0.59},
            {'x': 0.76, "y": 0.24}
        ], iis[1])
        self.assertEqual(['y', 'y', 'y', 'x'], gis[2])

    def test_analyze_collocation_bigram(self):
        # Input text for testing
        input_text = "I do not like green eggs and ham, I do not like them Sam I am!"

        # Expected collocations based on the input text
        expected_collocations = [(',', 'I'), ('I', 'am'), ('I', 'do'), ('Sam', 'I'), ('am', '!'),
                                 ('and', 'ham'), ('do', 'not'), ('eggs', 'and'), ('green', 'eggs'),
                                 ('ham', ','), ('like', 'green'), ('like', 'them'), ('not', 'like'),
                                 ('them', 'Sam')]

        # Perform collocation analysis
        result_collocations = get_collocation_ngram_score(input_text, gram=2)

        # Assert that the result matches the expected collocations
        score_ngrams = sorted(gram for gram, score in result_collocations)
        self.assertEqual(score_ngrams, expected_collocations)

    def test_analyze_collocation_trigram(self):
        # Input text for testing
        input_text = "I do not like green eggs and ham, I do not like them Sam I am!"

        # Expected collocations based on the input text
        expected_collocations = [('I', 'do', 'not'), ('do', 'not', 'like')]

        # Perform collocation analysis
        result_collocations = get_collocation_n_best(input_text, gram=3, n=2)

        # Assert that the result matches the expected collocations
        n_best = sorted(result_collocations)
        self.assertEqual(n_best, expected_collocations)

    def test_analyze_corpus(self):
        # Input corpus for testing
        corpus = [
            "This is a sample document. It contains various words.",
            "Another document with different words and punctuation!"
        ]

        # Expected result (dummy result for demonstration purposes)
        expected_result = [
            [('sample', 'JJ'), ('document', 'NN'), ('contains', 'VBZ'), ('various', 'JJ'),
             ('words', 'NNS')],
            [('another', 'DT'), ('document', 'NN'), ('different', 'JJ'), ('words', 'NNS'), ('punctuation', 'NN')]
        ]

        # Perform the analysis
        result = analyze_corpus(corpus)

        # Assert that the result matches the expected result
        self.assertEqual(result, expected_result)

    def test_generate_tree(self):
        # Input and (expected) output
        text = 'I rode an elephant in my pajamas'
        output = ['''(S (NP I) (VP (VP (V rode) (NP (Det an) (N elephant))) (PP (P in) (NP (Det my) (N pajamas)))))''',
                  '''(S (NP I) (VP (V rode) (NP (Det an) (N elephant) (PP (P in) (NP (Det my) (N pajamas))))))''']

        # Perform syntactic parsing
        grammar = nltk.CFG.fromstring("""
        S -> NP VP
        PP -> P NP
        NP -> Det N | Det N PP | 'I'
        VP -> V NP | VP PP
        Det -> 'an' | 'my'
        N -> 'elephant' | 'pajamas'
        V -> 'rode'
        P -> 'in'
        """)

        # For each tree generated due to differentiating sentence structure, confirm it is in the output above
        trees = generate_tree(text, grammar)
        for tree in trees:
            tree = ' '.join(str(tree).split())
            self.assertTrue(tree in output)


if __name__ == '__main__':
    unittest.main()
