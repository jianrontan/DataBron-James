from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL
import spacy
from typing import Dict, List, Tuple
import requests


class EntityLinker:
    def __init__(self, knowledge_base, batch_size: int = 10):
        self.WD = Namespace("http://www.wikidata.org/entity/")
        self.WDT = Namespace("http://www.wikidata.org/prop/direct/")
        self.SCHEMA = Namespace("http://schema.org/")

        self.graph = Graph()
        self.graph.bind("wd", self.WD)
        self.graph.bind("wdt", self.WDT)
        self.graph.bind("schema", self.SCHEMA)

        self.kb = knowledge_base
        self.batch_size = batch_size
        self.query_cache = {}
        self.nlp = spacy.load("en_core_web_sm")

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

    def link_entities(self, text: str, article_url: str) -> Dict:
        doc = self.nlp(text)

        entities = self._extract_and_link_entities(doc, article_url)

        relationships = self._extract_relationships(doc)

        return {
            'article_url': article_url,
            'entities': entities,
            'relationships': relationships
        }

    def _query_wikidata(self, entity_text: str) -> List[Dict]:
        """Query Wikidata for entity candidates"""
        query = f"""
        SELECT ?item ?itemLabel ?score WHERE {{
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:api "EntitySearch" .
                bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                bd:serviceParam mwapi:search "{entity_text}" .
                bd:serviceParam mwapi:language "en" .
                ?item wikibase:apiOutputItem mwapi:item .
                ?score wikibase:apiScore mwapi:score .
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }} ORDER BY DESC(?score) LIMIT 5
        """

        try:
            response = requests.get(
                'https://query.wikidata.org/sparql',
                params={'query': query, 'format': 'json'},
                headers={'User-Agent': 'NLPProject/1.0'}
            )
            results = response.json()
            return results['results']['bindings']
        except Exception as e:
            print(f"Wikidata query error: {e}")
            return []

    def _extract_and_link_entities(self, doc, article_url: str) -> List[Dict]:
        """Extract entities and link to Wikidata with confidence scores"""
        linked_entities = []
        total_entities = len(doc.ents)

        if total_entities > 0:
            print(f"\n  Found {total_entities} entities in article: {
                  article_url[:50]}...")

        for ent in doc.ents:
            candidates = self._query_wikidata(ent.text)
            if candidates:
                best_match = candidates[0]
                entity_uri = best_match['item']['value']
                confidence = float(best_match.get(
                    'score', {}).get('value', 0.5))
                entity_data = {
                    'text': ent.text,
                    'type': ent.label_,
                    'wikidata_uri': entity_uri,
                    'confidence': confidence
                }
                self.kb.add_entity(
                    text=ent.text,
                    entity_type=ent.label_,
                    uri=entity_uri,
                    confidence=confidence,
                    article_url=article_url
                )
                linked_entities.append(entity_data)
        return linked_entities

    def _extract_relationships(self, doc) -> List[Dict]:
        """Extract weighted relationships between entities"""
        relationships = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = self._find_subject(token)
                    obj = self._find_object(token)

                    if subject and obj:
                        confidence = self._calculate_relationship_confidence(
                            subject, token, obj)

                        rel_data = {
                            'subject': subject.text,
                            'predicate': token.text,
                            'object': obj.text,
                            'confidence': confidence
                        }

                        relationships.append(rel_data)

        return relationships

    def _calculate_relationship_confidence(
            self, subject, predicate, obj) -> float:
        """Calculate confidence score for relationship"""
        base_score = 0.5
        distance_penalty = 0.1 * (
            abs(predicate.i - subject.i) + abs(predicate.i - obj.i))

        confidence = max(0.1, min(1.0, base_score - distance_penalty))
        return confidence

    def process_batch(self, articles: List[Dict]):
        """Process articles in batches with proper progress tracking"""
        results = []
        total = len(articles)

        print(f"\nProcessing {
              total} articles for entities and relationships...")

        for i in range(0, len(articles), self.batch_size):
            batch = articles[i:i + self.batch_size]
            batch_results = []

            # Progress bar
            progress = (i + 1) / total * 100
            bar_length = 30
            filled_length = int(bar_length * progress // 100)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rProgress: [{bar}] {
                  progress:.1f}% - Processing batch {i+1}/{total}', end='')

            for article in batch:
                doc = self.nlp(article['text'])
                entities = self._extract_and_link_entities(
                    doc=doc,
                    article_url=article['url']
                )
                relationships = self._extract_relationships(doc)
                batch_results.append({
                    'article_url': article['url'],
                    'entities': entities,
                    'relationships': relationships
                })

            results.extend(batch_results)
            self._update_knowledge_base(batch_results)

        print("\nEntity processing complete!")
        print(f"Total entities found: {
              sum(len(r['entities']) for r in results)}")
        print(f"Total relationships found: {
              sum(len(r['relationships']) for r in results)}")

        return results

    def _update_knowledge_base(self, batch_results):
        """Update knowledge base with batch results"""
        for result in batch_results:
            for entity in result['entities']:
                self.kb.add_entity(
                    text=entity['text'],
                    entity_type=entity['type'],
                    uri=entity.get('wikidata_uri', ''),
                    confidence=entity.get('confidence', 0.5),
                    article_url=result['article_url']
                )

            for rel in result['relationships']:
                self.kb.add_relationship(
                    subject=rel['subject'],
                    predicate=rel['predicate'],
                    object=rel['object'],
                    confidence=rel.get('confidence', 0.5),
                    article_url=result['article_url']
                )
