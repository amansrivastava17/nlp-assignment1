# models.py

from sentiment_data import *
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_counts = Counter()
        for word in sentence:
            word_lower = word.lower()
            if add_to_indexer:
                word_idx = self.indexer.add_and_get_index(word_lower, add=True)
            else:
                word_idx = self.indexer.index_of(word_lower)
            if word_idx != -1:
                feature_counts[word_idx] += 1
        return feature_counts


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_counts = Counter()
        for word in sentence:
            word_lower = word.lower()
            if add_to_indexer:
                word_idx = self.indexer.add_and_get_index(word_lower, add=True)
            else:
                word_idx = self.indexer.index_of(word_lower)
            if word_idx != -1:
                feature_counts[word_idx] += 1
        
        for i in range(len(sentence) - 1):
            bigram = sentence[i].lower() + "_" + sentence[i+1].lower()
            if add_to_indexer:
                bigram_idx = self.indexer.add_and_get_index(bigram, add=True)
            else:
                bigram_idx = self.indexer.index_of(bigram)
            if bigram_idx != -1:
                feature_counts[bigram_idx] += 1
        return feature_counts


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                          'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                          'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                          'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i',
                          'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which',
                          'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                          'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'same',
                          'so', 'than', 'too', 'very', 'just', 'there'}
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_counts = Counter()
        
        for word in sentence:
            word_lower = word.lower()
            if word_lower not in self.stop_words:
                if add_to_indexer:
                    word_idx = self.indexer.add_and_get_index(word_lower, add=True)
                else:
                    word_idx = self.indexer.index_of(word_lower)
                if word_idx != -1:
                    feature_counts[word_idx] += 1
        
        for i in range(len(sentence) - 1):
            w1_lower = sentence[i].lower()
            w2_lower = sentence[i+1].lower()
            if w1_lower not in self.stop_words and w2_lower not in self.stop_words:
                bigram = w1_lower + "_" + w2_lower
                if add_to_indexer:
                    bigram_idx = self.indexer.add_and_get_index(bigram, add=True)
                else:
                    bigram_idx = self.indexer.index_of(bigram)
                if bigram_idx != -1:
                    feature_counts[bigram_idx] += 1
        
        sentence_lower = [w.lower() for w in sentence]
        negation_words = {'not', 'no', 'never', 'neither', 'nor', "n't", "not"}
        for neg_word in negation_words:
            if neg_word in sentence_lower:
                neg_feature = "HAS_NEGATION"
                if add_to_indexer:
                    neg_idx = self.indexer.add_and_get_index(neg_feature, add=True)
                else:
                    neg_idx = self.indexer.index_of(neg_feature)
                if neg_idx != -1:
                    feature_counts[neg_idx] = 1
                break
        
        positive_words = {'excellent', 'good', 'great', 'amazing', 'wonderful', 'best', 'love', 'fantastic', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing'}
        
        pos_count = sum(1 for w in sentence_lower if w in positive_words)
        neg_count = sum(1 for w in sentence_lower if w in negative_words)
        
        if pos_count > 0:
            pos_feature = "POS_WORDS_" + str(min(pos_count, 3))
            if add_to_indexer:
                pos_idx = self.indexer.add_and_get_index(pos_feature, add=True)
            else:
                pos_idx = self.indexer.index_of(pos_feature)
            if pos_idx != -1:
                feature_counts[pos_idx] = 1
        
        if neg_count > 0:
            neg_feature = "NEG_WORDS_" + str(min(neg_count, 3))
            if add_to_indexer:
                neg_idx = self.indexer.add_and_get_index(neg_feature, add=True)
            else:
                neg_idx = self.indexer.index_of(neg_feature)
            if neg_idx != -1:
                feature_counts[neg_idx] = 1
        
        return feature_counts


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor, weights):
        self.feat_extractor = feat_extractor
        self.weights = weights
    
    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor, weights):
        self.feat_extractor = feat_extractor
        self.weights = weights
    
    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        import math
        prob = 1.0 / (1.0 + math.exp(-score)) if score > -100 else 0.0
        return 1 if prob > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    num_features = len(feat_extractor.get_indexer())
    weights = [0.0] * num_features
    
    num_epochs = 10
    learning_rate = 0.1
    
    for epoch in range(num_epochs):
        num_correct = 0
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * count for idx, count in features.items())
            prediction = 1 if score > 0 else 0
            
            if prediction == ex.label:
                num_correct += 1
            else:
                if ex.label == 1:
                    for idx, count in features.items():
                        weights[idx] += learning_rate * count
                else:
                    for idx, count in features.items():
                        weights[idx] -= learning_rate * count
        
        accuracy = num_correct / len(train_exs)
        print(f"Epoch {epoch + 1}: Training accuracy = {accuracy:.3f}")
    
    return PerceptronClassifier(feat_extractor, weights)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    import math
    import random
    
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    num_features = len(feat_extractor.get_indexer())
    weights = [0.0] * num_features
    
    num_epochs = 20
    learning_rate = 0.1
    reg_lambda = 0.01
    
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        num_correct = 0
        
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * count for idx, count in features.items())
            
            if score > 100:
                prob = 1.0
            elif score < -100:
                prob = 0.0
            else:
                prob = 1.0 / (1.0 + math.exp(-score))
            
            prediction = 1 if prob > 0.5 else 0
            if prediction == ex.label:
                num_correct += 1
            
            error = ex.label - prob
            
            for idx, count in features.items():
                weights[idx] += learning_rate * (error * count - reg_lambda * weights[idx])
            
            if ex.label == 1:
                total_loss -= math.log(max(prob, 1e-10))
            else:
                total_loss -= math.log(max(1 - prob, 1e-10))
        
        accuracy = num_correct / len(train_exs)
        avg_loss = total_loss / len(train_exs)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.3f}, Training accuracy = {accuracy:.3f}")
    
    return LogisticRegressionClassifier(feat_extractor, weights)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model