import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define file paths
RAW_CSV_PATH = "data/raw_dreams.csv"
TRAIN_TXT_PATH = "data/train_dreams.txt"
VAL_TXT_PATH = "data/val_dreams.txt"

def preprocess_dream_dataset(input_csv_path, train_txt_path, val_txt_path, test_size=0.2):
    """
    Loads a dream interpretation dataset, splits it into training and validation sets,
    formats them for GPT-2 fine-tuning, and saves them as separate text files.
    """
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Ensure required columns exist
    if "Dream Symbol" not in df.columns or "Interpretation" not in df.columns:
        raise ValueError("Dataset must contain 'Dream Symbol' and 'Interpretation' columns.")

    # Convert dataset into formatted text
    formatted_texts = [f"Dream: {row['Dream Symbol']}\nInterpretation: {row['Interpretation']}\n" for _, row in df.iterrows()]

    # Split into train and validation sets
    train_texts, val_texts = train_test_split(formatted_texts, test_size=test_size, random_state=42)

    # Save training set
    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))

    # Save validation set
    with open(val_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_texts))

    print(f"✅ Dataset split successfully! Train: {len(train_texts)} samples, Validation: {len(val_texts)} samples")
    print(f"Train data saved to {train_txt_path}")
    print(f"Validation data saved to {val_txt_path}")

if __name__ == "__main__":
    print("🚀 Running Data Preparation...")
    preprocess_dream_dataset(RAW_CSV_PATH, TRAIN_TXT_PATH, VAL_TXT_PATH)
    print("✅ Data preparation complete.")
