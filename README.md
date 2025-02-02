# SMU BIA Datathon Natural Language Processing Pipeline

## Project Overview
This is a Natural Language Processing (NLP) pipeline that processes news articles, extracts entities, builds a knowledge base, performs feature engineering, and trains a model on the data. The system is designed to analyze text data, identify named entities, and create structured representations of the information.

## Technical Architecture
The project consists of several key components:
- Text Preprocessing Module
- Feature Engineering Module 
- Knowledge Base System
- Entity Linking System
- Data Processing Pipeline
- Relationship Classification Module

## Dependencies
- pandas
- spacy (en_core_web_sm model)
- nltk
- numpy
- rdflib
- sumy
- requests

## Installation
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas spacy nltk numpy rdflib sumy requests openpyxl

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Download spaCy model
python -m spacy download en_core_web_sm

## Project Structure
project/
├── main.py                    # Main execution script
├── src/
│   ├── preprocessing.py       # Text preprocessing module
│   ├── feature_engineering.py # Feature extraction module
│   ├── knowledge_base.py      # Knowledge base management
│   ├── entity_linker.py      # Entity linking system
│   ├── wikidata_client.py     # Wikidata API client
|   └── relationship_classifier.py     # Model training module
└── data/
    ├── raw/                   # Raw input data
    ├── processed/             # Processed data outputs
    └── features/             # Extracted features

## Pipeline Components

### Text Preprocessing
- Cleans and normalizes text data
- Removes special characters and stopwords
- Performs tokenization and sentence splitting

### Entity Linking
- Identifies named entities in text
- Links entities to Wikidata knowledge base
- Calculates confidence scores for entity matches

### Feature Engineering
- Extracts TF-IDF features
- Generates entity-based features
- Creates statistical text features
- Computes semantic embeddings

### Knowledge Base
- Stores entity and relationship information
- Maintains confidence scores
- Supports graph-based data structure

### Relationship Classifier
- Uses BERT for sequence classification
- Includes label encoding functionality
- Handles model training and saving

## Output Structure
data/processed/
├── preprocessed_news.csv      # Cleaned and preprocessed text
├── entity_analysis.csv        # Extracted entities
└── knowledge_base/           # Knowledge base data
data/features/
└── news_features.csv         # Extracted features

## Usage
# Run the main pipeline
python main.py

## Technical Notes
- Uses spaCy for NLP tasks and entity recognition
- Implements batch processing for efficient entity linking
- Supports caching for Wikidata queries
- Uses RDF graph structure for knowledge representation
- Implements confidence scoring for entity matches and relationships
