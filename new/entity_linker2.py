import spacy
from typing import Dict, List
from .knowledge_base import KnowledgeBase


class EntityLinker:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.nlp = spacy.load('en_core_web_sm')
        self.kb = knowledge_base

    def link_entities(self, text: str, article_url: str) -> Dict:
        """Link entities in text to knowledge base"""
        doc = self.nlp(text)

        # Extract and link entities
        entities = self._extract_and_link_entities(doc, article_url)

        # Extract relationships
        relationships = self._extract_relationships(doc)

        return {
            'article_url': article_url,
            'entities': entities,
            'relationships': relationships
        }

    def _extract_relationships(self, doc):
        """Extract relationships from doc"""
        relationships = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = self._find_subject(token)
                    obj = self._find_object(token)
                    if subject and obj:
                        relationships.append({
                            'subject': subject.text,
                            'predicate': token.text,
                            'object': obj.text
                        })
        return relationships

    def _find_subject(self, token):
        """Find the subject connected to the token"""
        for child in token.lefts:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child
        return None

    def _find_object(self, token):
        """Find the object connected to the token"""
        for child in token.rights:
            if child.dep_ in ["dobj", "pobj"]:
                return child
        return None

    def _extract_and_link_entities(self, doc, article_url):
        """Extract entities and add to knowledge base"""
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

            # Add to knowledge base
            self.kb.add_entity(ent.text, ent.label_, article_url)

        return entities
    
    def extract_relationships(self, text: str, article_url: str) -> Dict:
        """Extract and format relationships from text"""
        doc = self.nlp(text)
        relationships = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = self._find_subject(token)
                    obj = self._find_object(token)
                    if subject and obj:
                        relationships.append({
                            'subject': subject.text,
                            'predicate': token.text,
                            'object': obj.text
                        })
        return relationships
