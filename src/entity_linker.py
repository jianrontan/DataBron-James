from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL
from typing import Dict, List, Tuple
import requests
import time


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
        self.entity_cache = {}

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

    def _batch_query_wikidata(self, entity_texts: List[str]) -> Dict[str, List[Dict]]:
        """Query Wikidata for multiple entities at once but maintain individual results"""
        results = {}
        new_entities = [
            text for text in entity_texts if text not in self.entity_cache]

        if not new_entities:
            return {text: self.entity_cache[text] for text in entity_texts}

        chunk_size = 5
        for i in range(0, len(new_entities), chunk_size):
            chunk = new_entities[i:i + chunk_size]
            values = ' '.join(f'"{text}"' for text in chunk)

            query = f"""
            SELECT ?search ?item ?itemLabel ?score WHERE {{
                VALUES ?search {{ {values} }}
                SERVICE wikibase:mwapi {{
                    bd:serviceParam wikibase:api "EntitySearch" .
                    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                    bd:serviceParam mwapi:search ?search .
                    bd:serviceParam mwapi:language "en" .
                    ?item wikibase:apiOutputItem mwapi:item .
                    ?score wikibase:apiScore mwapi:score .
                }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }} ORDER BY ?search DESC(?score)
            """

            try:
                response = requests.get(
                    'https://query.wikidata.org/sparql',
                    params={'query': query, 'format': 'json'},
                    headers={'User-Agent': 'NLPProject/1.0'}
                )
                data = response.json()

                for binding in data['results']['bindings']:
                    search_term = binding['search']['value']
                    if search_term not in results:
                        results[search_term] = []
                    results[search_term].append(binding)
                    self.entity_cache[search_term] = results[search_term]

                time.sleep(0.1)

            except Exception as e:
                print(f"Wikidata batch query error: {e}")

        return {text: self.entity_cache.get(text, []) for text in entity_texts}

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

    def _calculate_relationship_confidence(self, subject, predicate, obj) -> float:
        """Calculate confidence score for relationship using dependency information"""
        base_score = 0.5

        dep_bonus = 0.0
        if subject.dep_ in ["nsubj", "nsubjpass"]:
            dep_bonus += 0.2
        if obj.dep_ in ["dobj", "pobj"]:
            dep_bonus += 0.2

        distance_penalty = 0.05 * (
            abs(predicate.i - subject.i) + abs(predicate.i - obj.i))

        confidence = max(0.1, min(1.0, base_score +
                         dep_bonus - distance_penalty))
        return confidence

    def _process_entity_candidate(self, entity_text: str, entity_type: str, candidate: Dict) -> Dict:
        """
        Process a Wikidata entity candidate and create entity data structure.

        Args:
            entity_text (str): Original entity text from the document
            entity_type (str): NER type of the entity
            candidate (Dict): Best matching Wikidata candidate

        Returns:
            Dict: Processed entity data with URI and confidence score
        """
        try:
            # Extract Wikidata URI
            entity_uri = candidate['item']['value']

            # Get confidence score from Wikidata match
            confidence = float(candidate.get('score', {}).get('value', 0.5))

            # Create entity data structure
            entity_data = {
                'text': entity_text,
                'type': entity_type,
                'wikidata_uri': entity_uri,
                'confidence': confidence
            }

            return entity_data

        except Exception as e:
            print(f"Error processing entity candidate: {e}")
            # Return default entity data if processing fails
            return {
                'text': entity_text,
                'type': entity_type,
                'wikidata_uri': '',
                'confidence': 0.1
            }

    def _extract_and_link_entities_from_preprocessed(self, text: str, tokenized: list, ner: list, article_url: str):
        """Extract and link entities from preprocessed data"""
        linked_entities = []
        total_entities = len(ner)
        print(f"\n  Found {total_entities} entities in article: {
              article_url[:50]}...")

        entity_texts = [entity_text for entity_text, _ in ner]

        entity_results = self._batch_query_wikidata(entity_texts)

        for entity_text, entity_type in ner:
            candidates = entity_results.get(entity_text, [])
            if candidates:
                entity_data = self._process_entity_candidate(
                    entity_text,
                    entity_type,
                    candidates[0]
                )
                linked_entities.append(entity_data)

        return linked_entities

    def _extract_relationships_from_tokenized(self, doc):
        """Extract relationships using preprocessed dependencies"""
        relationships = []

        dep_map = {token: {'lefts': list(token.lefts), 'rights': list(token.rights)}
                   for token in doc}

        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = next((child for child in dep_map[token]['lefts']
                                    if child.dep_ in ["nsubj", "nsubjpass"]), None)
                    obj = next((child for child in dep_map[token]['rights']
                                if child.dep_ in ["dobj", "pobj"]), None)

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

    def process_batch(self, articles: List[Dict]):
        """Process articles in batches with proper progress tracking"""
        results = []
        total = len(articles)
        print(f"\nProcessing {
              total} articles for entities and relationships...")

        for i, article in enumerate(articles):
            entities = self._extract_and_link_entities_from_preprocessed(
                text=article['text'],
                tokenized=article['tokenized'],
                ner=article['ner'],
                article_url=article['url']
            )

            relationships = self._extract_relationships_from_tokenized(
                article['doc']
            )

            article_result = {
                'article_url': article['url'],
                'entities': entities,
                'relationships': relationships
            }

            results.append(article_result)
            self._update_knowledge_base([article_result])

            # Update progress bar
            progress = (i + 1) / total * 100
            bar_length = 30
            filled_length = int(bar_length * progress // 100)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rProgress: [{bar}] {
                  progress:.1f}% - Processing article {i+1}/{total}', end='')

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
