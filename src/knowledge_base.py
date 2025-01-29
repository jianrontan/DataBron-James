import networkx as nx
import rdflib
import pandas as pd
from typing import Dict, List


class KnowledgeBase:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {
            'PERSON': {},      # People, including fictional
            'NORP': {},        # Nationalities or religious or political groups
            'FAC': {},         # Buildings, airports, highways, bridges, etc.
            'ORG': {},         # Companies, agencies, institutions, etc.
            'GPE': {},         # Countries, cities, states
            'LOC': {},         # Non-GPE locations, mountain ranges, bodies of water
            'PRODUCT': {},     # Objects, vehicles, foods, etc.
            # Named hurricanes, battles, wars, sports events, etc.
            'EVENT': {},
            'WORK_OF_ART': {},  # Titles of books, songs, etc.
            'LAW': {},         # Named documents made into laws
            'LANGUAGE': {},    # Any named language
            'DATE': {},        # Absolute or relative dates or periods
            'TIME': {},        # Times smaller than a day
            'PERCENT': {},     # Percentage
            'MONEY': {},       # Monetary values, including unit
            'QUANTITY': {},    # Measurements, as of weight or distance
            'ORDINAL': {},     # "first", "second", etc.
            'CARDINAL': {}     # Numerals that don't fall under another type
        }

    def add_entity(self, text: str, entity_type: str, article_url: str):
        """Add entity to knowledge base"""
        # Add node to graph
        node_id = f"{entity_type}_{text}"
        self.graph.add_node(
            node_id,
            type=entity_type,
            text=text,
            mentions=[article_url]
        )

        # Add to entity dictionary
        if text not in self.entities[entity_type]:
            self.entities[entity_type][text] = {
                'mentions': [],
                'relationships': []
            }

        if article_url not in self.entities[entity_type][text]['mentions']:
            self.entities[entity_type][text]['mentions'].append(article_url)

    def add_relationship(self, subject: str, predicate: str, object: str):
        """Add relationship between entities"""
        self.graph.add_edge(subject, object, relation=predicate)

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
