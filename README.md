# Fine-Tuning GPT-2 for Freud-Based Dream Interpretation

## Project Overview
This project explores the fine-tuning of GPT-2 models for Freudian dream interpretation. We trained two different sizes of GPT-2 (Small - 124M, Medium - 355M) on a structured dataset derived from Kaggle's Dictionary of Dreams and evaluated them against Freud-style dream analysis.

### Main Objectives:
- Assess whether fine-tuning improves GPT-2’s ability to interpret dreams.
- Compare GPT-2 Small vs. GPT-2 Medium to analyze the effect of model size.
- Use BLEU, ROUGE, Perplexity, and BERTScore to evaluate performance.

### Key Findings:
- Fine-tuning significantly improved Freud-style interpretations.
- GPT-2 Medium (Fine-Tuned) outperformed all other models.
- GPT-2 Small (Fine-Tuned) had high accuracy but overfitting issues.


## Installation & Setup

### 1. Install Dependencies
Ensure Python 3.8+ is installed. Install required libraries:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Run the following command to process the dataset:
```bash
python scripts/data_prep.py
```
This will generate structured training and validation datasets in `/data/`.

### 3. Train the Model
Fine-tune GPT-2 Small or GPT-2 Medium with:
```bash
python scripts/train_gpt2.py  # Train GPT-2 Small
python scripts/train_gpt2_medium_cpu.py  # Train GPT-2 Medium
```
Training automatically saves checkpoints to allow resuming in case of interruptions.

### 4. Evaluate the Model
To compare models using BLEU, ROUGE, Perplexity, and BERTScore:
```bash
python scripts/evaluate_model_updated.py
```
Results are saved in `/results/`.

### 5. Test the Model with Your Own Dreams
Run an interactive script to test the trained models:
```bash
python scripts/test_finetuned_gpt2.py --model models/fine_tuned_gpt2_medium_cpu
```
Example input/output:
```
Enter your dream: I was flying over a city.
Interpretation: To dream that you are flying represents a desire for freedom and escape from constraints.
```

## Model Performance Summary
| Model                      | BLEU (↑) | ROUGE (↑) | Perplexity (↓) | BERTScore (↑) |
|----------------------------|----------|-----------|----------------|---------------|
| GPT-2 Small (Original)     | 0.066    | 0.212     | 21,296         | 0.518         |
| GPT-2 Small (Fine-Tuned)   | 0.121    | 0.317     | 1.77B (unstable)| 0.659        |
| GPT-2 Medium (Original)    | 0.066    | 0.229     | 263,059        | 0.525         |
| GPT-2 Medium (Fine-Tuned)  | 0.099    | 0.308     | 41M            | 0.651         |

### Key Takeaways:
- Fine-tuning improves dream interpretation accuracy.
- GPT-2 Medium (Fine-Tuned) performs best, balancing fluency and accuracy.
- GPT-2 Small (Fine-Tuned) is less stable due to overfitting.


## Future Work
- Extend fine-tuning to larger LLMs (DeepSeek, LLaMA, GPT-3.5 Turbo).
- Test models on zero-shot dream interpretations.
- Improve dataset size to include more diverse dream types.

## Contact & Contributions
Feel free to submit pull requests or reach out for collaborations.

## Special Thanks

A thank you to Dr. Sharon Yalov-Handzel for her guidance, support, and insights throughout this project.
