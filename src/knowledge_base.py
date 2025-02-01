import networkx as nx
import pandas as pd
from typing import Dict, List
from .wikidata_client import WikidataClient


class KnowledgeBase:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {
            'PERSON': {},      # People, including fictional
            'NORP': {},        # Nationalities, religious, political groups
            'FAC': {},         # Buildings, airports, highways, bridges
            'ORG': {},         # Companies, agencies, institutions
            'GPE': {},         # Countries, cities, states
            'LOC': {},         # Non-GPE locations, mountains, water bodies
            'PRODUCT': {},     # Objects, vehicles, foods
            'EVENT': {},       # Named hurricanes, battles, wars, sports events
            'WORK_OF_ART': {},  # Titles of books, songs
            'LAW': {},         # Named documents made into laws
            'LANGUAGE': {}     # Any named language
        }
        self.attributes = {
            'DATE': {},        # Absolute or relative dates or periods
            'TIME': {},        # Times smaller than a day
            'PERCENT': {},     # Percentage
            'MONEY': {},       # Monetary values, including unit
            'QUANTITY': {},    # Measurements, weight or distance
            'ORDINAL': {},     # "first", "second"
            'CARDINAL': {},    # Numerals
            'DURATION': {},    # Time durations
            'AGE': {},         # Age values
            'FREQUENCY': {},   # How often something occurs
            'SCORE': {},       # Numeric scores
            'STAT': {},       # Statistics and measurements
            'RATING': {},      # Ratings and rankings
            'DIMENSION': {}    # Physical dimensions
        }
        self.wikidata = WikidataClient()

    def add_entity(self, text: str, entity_type: str, uri: str, confidence: float, article_url: str):
        """Add entity with metadata"""
        node_id = uri if uri else f"{entity_type}_{text}"
        category = 'entities' if entity_type in self.entities else 'attributes'
        
        # Initialize dictionary entry if it doesn't exist
        target_dict = self.entities if category == 'entities' else self.attributes
        if entity_type not in target_dict:
            target_dict[entity_type] = {}
        if text not in target_dict[entity_type]:
            target_dict[entity_type][text] = {
                'uri': uri,
                'mentions': [article_url],
                'confidence': confidence,
                'confidence_counts': 1
            }
            # Add node to graph
            self.graph.add_node(
                node_id,
                type=entity_type,
                text=text,
                category=category,
                articles={article_url: confidence},
                mentions=1
            )
        else:
            # Update existing entry
            current_item = target_dict[entity_type][text]
            count = current_item.get('confidence_counts', 1)
            current_conf = current_item.get('confidence', confidence)
            new_conf = (current_conf * count + confidence) / (count + 1)
            
            # Update values
            current_item['confidence'] = new_conf
            current_item['confidence_counts'] = count + 1
            current_item['mentions'].append(article_url)

    def add_relationship(self, subject: str, predicate: str, object: str,
                         confidence: float = 0.5, article_url: str = None):
        """Add relationship between entities with confidence score"""
        self.graph.add_edge(
            subject,
            object,
            relation=predicate,
            confidence=confidence,
            article_url=article_url
        )

    def save(self, output_dir: str):
        """Save knowledge base"""
        try:
            # Save graph structure (with proper attribute handling)
            graph_file = f"{output_dir}/knowledge_graph.gexf"
            # Clean up node attributes before saving
            for node in self.graph.nodes():
                mentions = self.graph.nodes[node].get('mentions', [])
                if isinstance(mentions, list):
                    self.graph.nodes[node]['mentions'] = ','.join(mentions)

            nx.write_gexf(self.graph, graph_file)
            print(f"Saved knowledge graph to {graph_file}")

            # Save entity dictionary
            for entity_type, entities in self.entities.items():
                if entities:  # Only save if there are entities
                    file_path = f"{output_dir}/{entity_type}_entities.csv"
                    df = pd.DataFrame.from_dict(entities, orient='index')
                    df.to_csv(file_path)
                    print(f"Saved {entity_type} entities to {file_path}")

        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
