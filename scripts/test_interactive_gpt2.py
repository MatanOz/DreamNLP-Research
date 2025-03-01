import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
MODEL_PATH = "models/fine_tuned_gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_interpretation(dream_text, max_length=100):
    """
    Generate a dream interpretation using the fine-tuned GPT-2 model.
    """
    input_text = f"Dream: {dream_text}\nInterpretation:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output with improved settings
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        temperature=0.7, 
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids)
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Interactive loop for user input
print("🚀 Welcome to the Dream Interpreter!")
print("Type your dream below and press Enter to get an interpretation.")
print("Type 'exit' to quit.\n")

while True:
    user_dream = input("🛌 Your Dream: ")
    if user_dream.lower() == "exit":
        print("Goodbye! 💤")
        break

    interpretation = generate_interpretation(user_dream)
    print("\n🔮 Interpretation:", interpretation, "\n")
