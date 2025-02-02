import os
import pandas as pd
import torch
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ✅ Step 1: Set File Paths
base_dir = r"C:/Users/sooch/OneDrive/Desktop/DataBron-James/data"
raw_data_path = os.path.join(base_dir, "raw/news_features.csv")

train_csv = os.path.join(base_dir, "processed/train_data.csv")
train_csv_small = os.path.join(base_dir, "processed/train_data_small.csv")
valid_csv = os.path.join(base_dir, "processed/valid_data.csv")
test_csv = os.path.join(base_dir, "processed/test_data.csv")

os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)


# ✅ Step 2: Load, Rename & Split Data
def split_and_save_data(csv_file, train_csv, train_csv_small, valid_csv, test_csv):
    """ Load, rename columns if needed, split into train/valid/test, and save. """
    if not os.path.exists(csv_file):
        print(f"❌ Error: Missing file {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)

        # ✅ Print original column names for debugging
        print("✅ Original Columns in Dataset:", df.columns.tolist())

        # ✅ Remove spaces and convert column names to lowercase
        df.columns = df.columns.str.strip().str.lower()

        # ✅ Rename columns to match expected format
        column_mapping = {
            "subject": "entity1",
            "relation": "predicate",
            "object": "entity2",
            "head": "entity1",
            "tail": "entity2",
            "relationtype": "predicate",
        }
        df.rename(columns=column_mapping, inplace=True)

        # ✅ Print updated column names
        print("✅ Updated Columns in Dataset:", df.columns.tolist())

        # ✅ Ensure required columns exist
        required_columns = {"entity1", "predicate", "entity2"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"❌ Missing required columns in dataset: {missing_columns}")

        # ✅ Remove rows where `entity1` is NaN
        df = df.dropna(subset=["entity1"])

        # ✅ Count occurrences of each class
        predicate_counts = Counter(df["predicate"])
        print("✅ Class distribution:", predicate_counts)

        # ✅ Remove classes that have fewer than 2 samples
        df = df[df["predicate"].map(predicate_counts) > 1]

        if df.empty:
            raise ValueError("❌ Error: After filtering, no data is left. Ensure dataset has at least 2 samples per class.")

        # ✅ Attempt stratified split
        try:
            train_data, temp_data = train_test_split(df, test_size=0.2, stratify=df["predicate"], random_state=42)
            valid_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["predicate"], random_state=42)
        except ValueError:
            print("⚠️ Warning: Not enough samples for stratified split. Using random split instead.")
            train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
            valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # ✅ Directly Sample 5% of `train_data`
        train_data_small = train_data.sample(frac=0.05, random_state=42)

        # ✅ Save datasets
        train_data.to_csv(train_csv, index=False)
        train_data_small.to_csv(train_csv_small, index=False)
        valid_data.to_csv(valid_csv, index=False)
        test_data.to_csv(test_csv, index=False)

        print(f"✅ Data split & saved: {train_csv}, {train_csv_small}, {valid_csv}, {test_csv}")
        return train_data

    except Exception as e:
        print(f"❌ Error processing {csv_file}: {str(e)}")
        return None


# ✅ Step 3: Run Data Split
train_data = split_and_save_data(raw_data_path, train_csv, train_csv_small, valid_csv, test_csv)


# ✅ Step 4: Check If All CSV Files Exist
for file in [train_csv, train_csv_small, valid_csv, test_csv]:
    if os.path.exists(file):
        print(f"✅ CSV file ready: {file}")
    else:
        print(f"❌ Missing CSV file: {file}")


# ✅ Step 5: Define Dataset Class for BERT
class RelationshipDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # ✅ Convert to string to avoid NaN/float errors
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),  # ✅ Change dtype to float
        }


# ✅ Step 6: Define Relationship Classifier
class RelationshipClassifier:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = LabelEncoder()
        self.model = None  

    def train(self, train_path=train_csv_small, valid_path=valid_csv, output_dir="models/"):
        """Train BERT for relationship classification"""
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            raise FileNotFoundError("❌ Training or validation data files not found!")

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)

        # ✅ Remove NaN values before training
        train_df = train_df.dropna(subset=["entity1"])
        valid_df = valid_df.dropna(subset=["entity1"])

        required_columns = {"entity1", "predicate", "entity2"}
        missing_columns = required_columns - set(train_df.columns)
        if missing_columns:
            raise KeyError(f"❌ Missing required columns in training data: {missing_columns}")

        self.encoder.fit(train_df["predicate"])

        train_df["Label"] = self.encoder.transform(train_df["predicate"])
        valid_df["Label"] = self.encoder.transform(valid_df["predicate"])

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.encoder, os.path.join(output_dir, "label_encoder.pkl"))

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(self.encoder.classes_))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_dataset = RelationshipDataset(train_df["entity1"].tolist(), train_df["Label"].tolist(), self.tokenizer)
        valid_dataset = RelationshipDataset(valid_df["entity1"].tolist(), valid_df["Label"].tolist(), self.tokenizer)

        training_args = TrainingArguments(output_dir=output_dir, eval_strategy="epoch", per_device_train_batch_size=4, per_device_eval_batch_size=4, num_train_epochs=3, weight_decay=0.01)

        trainer = Trainer(model=self.model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)

        trainer.train()
        trainer.save_model(output_dir)
        print("✅ Model trained successfully.")

if __name__ == "__main__":
    classifier = RelationshipClassifier()
    classifier.train()
