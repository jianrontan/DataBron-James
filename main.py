import pandas as pd
from src.preprocessing import TextPreprocessor
from src.feature_engineering import TextFeatureEngineer


def main():
    """Main execution function"""
    try:
        # Load data
        df = pd.read_excel('data/news_excerpts_parsed.xlsx')

        # Initialize processors
        preprocessor = TextPreprocessor()
        feature_engineer = TextFeatureEngineer()

        # Process pipeline
        print("Starting preprocessing...")
        news_data = preprocessor.process_text('data/news_excerpts_parsed.xlsx')
        if news_data is not None:
            print("News data processed successfully")
            print(news_data.head())

        print("Extracting features...")
        features = feature_engineer.extract_features(news_data['Text'])

        print("Pipeline completed successfully")
        return features

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
