import pandas as pd
import os
from src.preprocessing import TextPreprocessor
from src.feature_engineering import TextFeatureEngineer

def create_output_directories():
    """Create output directories if they don't exist"""
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)

def main():
    """Main execution function"""
    try:
        create_output_directories()
        
        # Initialize processors
        preprocessor = TextPreprocessor()
        feature_engineer = TextFeatureEngineer()

        # Process pipeline
        print("Starting preprocessing...")
        news_data = preprocessor.process_text(
            'data/raw/news_excerpts_parsed.xlsx')

        if news_data is not None:
            print("News data processed successfully")
            
            preprocessed_file = 'data/processed/preprocessed_news.csv'
            news_data.to_csv(preprocessed_file, index=False, encoding='utf-8')
            print(f"Preprocessed data saved to {preprocessed_file}")
            
            print("\nPreprocessed Data Sample:")
            print(news_data[['Text', 'Cleaned_words', 'NER_Entities']].head())

            print("\nExtracting features...")
            features = feature_engineer.extract_features(news_data['Text'])

            if features is not None:
                features_file = 'data/features/news_features.csv'
                features.to_csv(features_file, index=False, encoding='utf-8')
                print(f"Features saved to {features_file}")
                
                print("\nFeature Engineering Complete")
                print(f"Feature Shape: {features.shape}")
                print("\nFeature Sample:")
                print(features.head())
                
                return features

            return features

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
