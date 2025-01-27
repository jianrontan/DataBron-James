import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy

class TextFeatureEngineer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer()
        self.count_vec = CountVectorizer()
        
    def extract_features(self, texts):
        """Extract all features from a list of texts"""
        features = {}
        
        # TF-IDF features
        features['tfidf'] = self._get_tfidf_features(texts)
        
        # Entity-based features
        features['entity'] = self._get_entity_features(texts)
        
        # Statistical features
        features['statistical'] = self._get_statistical_features(texts)
        
        # Combine all features
        return self._combine_features(features)
    
    def _get_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf.fit_transform(texts)
    
    def _get_entity_features(self, texts):
        """Extract named entity features"""
        entity_features = []
        for text in texts:
            doc = self.nlp(text)
            features = {
                'num_entities': len(doc.ents),
                'num_persons': len([e for e in doc.ents if e.label_ == 'PERSON']),
                'num_orgs': len([e for e in doc.ents if e.label_ == 'ORG']),
                'num_money': len([e for e in doc.ents if e.label_ == 'MONEY'])
            }
            entity_features.append(features)
        return pd.DataFrame(entity_features)
    
    def _get_statistical_features(self, texts):
        """Extract statistical features"""
        stats_features = []
        for text in texts:
            doc = self.nlp(text)
            features = {
                'text_length': len(text),
                'num_sentences': len(list(doc.sents)),
                'avg_word_length': np.mean([len(token.text) for token in doc]),
                'num_tokens': len(doc)
            }
            stats_features.append(features)
        return pd.DataFrame(stats_features)
    
    def _combine_features(self, features):
        """Combine all features into a single matrix"""
        # Convert sparse matrix to dense for TF-IDF
        tfidf_dense = pd.DataFrame(
            features['tfidf'].toarray(),
            columns=self.tfidf.get_feature_names_out()
        )
        
        # Combine with other features
        combined = pd.concat([
            tfidf_dense,
            features['entity'],
            features['statistical']
        ], axis=1)
        
        return combined