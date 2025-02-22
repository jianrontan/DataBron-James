import pandas as pd
import os
from src.preprocessing import TextPreprocessor
from src.feature_engineering import TextFeatureEngineer
from src.knowledge_base import KnowledgeBase
from src.entity_linker import EntityLinker
# from src.relationship_classifier import RelationshipClassifier


def create_output_directories():
    """Create output directories if they don't exist"""
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)


def save_news_data(news_data):
    try:
        os.makedirs('data/processed/knowledge_base', exist_ok=True)
        os.makedirs('data/features', exist_ok=True)

        if news_data is not None:
            preprocessed_file = 'data/processed/preprocessed_news.csv'
            news_data.to_csv(preprocessed_file, index=False, encoding='utf-8')
            print(f"Preprocessed data saved to {preprocessed_file}")

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()


def save_pipeline_results(news_data, all_entities, knowledge_base, features, relationships):
    """Save all pipeline results to appropriate files"""
    try:
        # Create directories
        os.makedirs('data/processed/knowledge_base', exist_ok=True)
        os.makedirs('data/features', exist_ok=True)

        # Save preprocessed data
        if news_data is not None:
            preprocessed_file = 'data/processed/preprocessed_news.csv'
            news_data.to_csv(preprocessed_file, index=False, encoding='utf-8')
            print(f"Preprocessed data saved to {preprocessed_file}")

        # Save entity analysis
        if all_entities:
            entity_file = 'data/processed/entity_analysis.csv'
            pd.DataFrame.from_records(all_entities).to_csv(
                entity_file, index=False, encoding='utf-8')
            print(f"Entity analysis saved to {entity_file}")

        # Save knowledge base
        if knowledge_base:
            kb_dir = 'data/processed/knowledge_base'
            knowledge_base.save(kb_dir)

        # Save features
        if features is not None:
            features_file = 'data/features/news_features.csv'
            features.to_csv(features_file, index=False, encoding='utf-8')
            print(f"Features saved to {features_file}")
        
        # Save relationships
        # if relationships is not None:
        #     relationships_file = 'data/processed/relationships.csv'
        #     pd.DataFrame(relationships).to_csv(
        #         relationships_file, index=False, encoding='utf-8')
        #     print(f"Relationships saved to {relationships_file}")

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function"""
    try:
        create_output_directories()

        # Initialize processors
        preprocessor = TextPreprocessor()
        feature_engineer = TextFeatureEngineer()
        knowledge_base = KnowledgeBase()
        entity_linker = EntityLinker(knowledge_base, batch_size=20)
        # classifier = RelationshipClassifier()

        # Process pipeline
        print("Starting preprocessing...")
        news_data = preprocessor.process_text(
            'data/raw/news_excerpts_parsed.xlsx')

        save_news_data(news_data)

        if news_data is not None:
            print("Processing entities and relationships...")
            articles = [
                {
                    'text': row['Cleaned_words'],
                    'tokenized': row['Tokenized'],
                    'ner': row['NER_Entities'],
                    'url': row['Link'],
                    'doc': row['doc']
                }
                for _, row in news_data.iterrows()
            ]

            all_entities = entity_linker.process_batch(articles)

            print("\nExtracting features...")
            features = feature_engineer.extract_features(
                news_data['Cleaned_words'])

            save_pipeline_results(news_data, all_entities,
                                  knowledge_base, features)

            return {
                'entities': all_entities,
                'knowledge_base': knowledge_base,
                'features': features
            }

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
