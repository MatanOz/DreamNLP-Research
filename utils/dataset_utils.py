import pandas as pd

def preprocess_dream_dataset(input_csv_path, output_txt_path):
    """
    Loads a dream interpretation dataset, formats it for GPT-2 fine-tuning,
    and saves it as a plain text file.
    """
    df = pd.read_csv(input_csv_path)
    if "Dream Symbol" not in df.columns or "Interpretation" not in df.columns:
        raise ValueError("Dataset must contain 'Dream Symbol' and 'Interpretation' columns.")
    
    formatted_texts = []
    for _, row in df.iterrows():
        dream = row["Dream Symbol"]
        interpretation = row["Interpretation"]
        formatted_text = f"Dream: {dream}\nInterpretation: {interpretation}\n"
        formatted_texts.append(formatted_text)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_texts))

    print(f"✅ Dataset processed successfully! Saved to {output_txt_path}")
