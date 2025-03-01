import torch
import pandas as pd
import evaluate
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Model selection
def load_model(model_path="gpt2"):
    """ Load a pre-trained or fine-tuned GPT-2 model """
    print(f"🔄 Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Fix padding token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Generate interpretation using the model
def generate_interpretation(model, tokenizer, dream_text):
    """ Generate a dream interpretation """
    input_text = f"Dream: {dream_text}\nInterpretation:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids, 
        max_length=50, 
        temperature=0.5, 
        top_p=0.9, 
        repetition_penalty=1.2, 
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids)
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Compute Perplexity
def calculate_perplexity(model, tokenizer, text):
    """ Compute perplexity of generated text """
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

# Evaluate model
def evaluate_model(model_path="models/fine_tuned_gpt2", test_data_path="data/freud_50_dreams.csv"):
    """ Evaluate model using BLEU, ROUGE, Perplexity, and BERTScore """
    model, tokenizer = load_model(model_path)
    df = pd.read_csv(test_data_path)

    # Adjust test set format for better alignment with model outputs
    df["Freudian Interpretation"] = df["Freudian Interpretation"].apply(lambda x: x.strip().lower())

    results = {"Dream": [], "Freud Interpretation": [], "Model Interpretation": [], "BLEU": [], "ROUGE": [], "Perplexity": [], "BERTScore": []}

    for _, row in df.iterrows():
        dream = row["Dream"]
        freud_interpretation = row["Freudian Interpretation"]
        model_interpretation = generate_interpretation(model, tokenizer, dream).lower()

        # BLEU Score
        reference = [freud_interpretation.split()]
        candidate = model_interpretation.split()
        bleu_score = sentence_bleu(reference, candidate)
        
        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_score = scorer.score(freud_interpretation, model_interpretation)["rougeL"].fmeasure

        # Perplexity
        perplexity = calculate_perplexity(model, tokenizer, dream)

        # BERTScore
        bert_score = bertscore.compute(predictions=[model_interpretation], references=[freud_interpretation], model_type="bert-base-uncased")["f1"][0]

        # Store results
        results["Dream"].append(dream)
        results["Freud Interpretation"].append(freud_interpretation)
        results["Model Interpretation"].append(model_interpretation)
        results["BLEU"].append(bleu_score)
        results["ROUGE"].append(rouge_score)
        results["Perplexity"].append(perplexity)
        results["BERTScore"].append(bert_score)

    # Save results to CSV
    output_path = f"evaluation_results_{model_path.replace('/', '_')}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    print(f"✅ Evaluation complete! Results saved to {output_path}")

# Run evaluation
if __name__ == "__main__":
    print("🚀 Starting evaluation...")
    model_to_test = input("Enter the model path (default: fine-tuned GPT-2) or type 'gpt2' for the base model: ") or "models/fine_tuned_gpt2"
    evaluate_model(model_to_test)
