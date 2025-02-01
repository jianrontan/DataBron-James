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
        self.svd = TruncatedSVD(n_components=100)

    def extract_features(self, texts):
        """Extract all features from a list of texts"""
        features = {}

        features['tfidf'] = self._get_tfidf_features(texts)
        features['entity'] = self._get_entity_features(texts)
        features['statistical'] = self._get_statistical_features(texts)
        features['semantic'] = self._get_semantic_features(texts)

        return self._combine_features(features)

    def _get_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf.fit_transform(texts)

    def _get_entity_features(self, texts):
        """Extract named entity features"""
        entity_features = []
        for text in texts:
            doc = self.nlp(text)
            entity_counts = {
                'num_entities': len(doc.ents),
                'num_persons': len([e for e in doc.ents if e.label_ == 'PERSON']),
                'num_orgs': len([e for e in doc.ents if e.label_ == 'ORG']),
                'num_money': len([e for e in doc.ents if e.label_ == 'MONEY']),
                'num_dates': len([e for e in doc.ents if e.label_ == 'DATE']),
                'num_locations': len([e for e in doc.ents if e.label_ == 'GPE' or e.label_ == 'LOC']),
                'num_events': len([e for e in doc.ents if e.label_ == 'EVENT'])
            }

            entity_counts.update({
                'unique_entity_types': len(set(ent.label_ for ent in doc.ents)),
                'entity_density': len(doc.ents) / len(doc) if len(doc) > 0 else 0,
                'person_ratio': entity_counts['num_persons'] / entity_counts['num_entities'] if entity_counts['num_entities'] > 0 else 0,
                'org_ratio': entity_counts['num_orgs'] / entity_counts['num_entities'] if entity_counts['num_entities'] > 0 else 0
            })

            entity_features.append(entity_counts)
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
                'num_tokens': len(doc),

                'unique_words': len(set([token.text.lower() for token in doc if not token.is_punct])),
                'stopwords_ratio': len([token for token in doc if token.is_stop]) / len(doc) if len(doc) > 0 else 0,
                'punct_ratio': len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0,

                'noun_phrases': len(list(doc.noun_chunks)),
                'verb_count': len([token for token in doc if token.pos_ == "VERB"]),
                'noun_count': len([token for token in doc if token.pos_ == "NOUN"]),
                'adj_count': len([token for token in doc if token.pos_ == "ADJ"]),

                'avg_sentence_length': np.mean([len([token for token in sent])
                                                for sent in doc.sents]) if len(list(doc.sents)) > 0 else 0
            }
            stats_features.append(features)
        return pd.DataFrame(stats_features)

    def _get_semantic_features(self, texts):
        """Extract semantic features using embeddings"""
        semantic_features = []
        for text in texts:
            doc = self.nlp(text)
            if doc.vector_norm:
                vector = doc.vector / doc.vector_norm
            else:
                vector = doc.vector

            features = {
                'embedding_mean': float(vector.mean()),
                'embedding_std': float(vector.std()),
                'embedding_norm': float(doc.vector_norm)
            }
            semantic_features.append(features)
        return pd.DataFrame(semantic_features)

    def _combine_features(self, features):
        """Combine all features with dimensionality reduction"""
        tfidf_reduced = pd.DataFrame(
            self.svd.fit_transform(features['tfidf']),
            columns=[f'tfidf_comp_{i}' for i in range(self.svd.n_components)]
        )

        combined = pd.concat([
            tfidf_reduced,
            features['entity'],
            features['statistical'],
            features['semantic']
        ], axis=1)

        return combined
