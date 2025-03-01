import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the original GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix padding issue
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

def generate_interpretation(dream_text, max_length=100):
    """
    Generate a dream interpretation using the original GPT-2 model.
    """
    input_text = f"Dream: {dream_text} Interpretation:\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output with improved settings
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        temperature=0.9,  # Higher temperature = more randomness
        top_p=0.95,  # Nucleus sampling
        repetition_penalty=1.2,  # Avoid repeating phrases
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids)  # Fix attention mask warning
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example dream prompts
dreams = [
    "I was chased by a monster",
    "I was swimming in a dark ocean",
    "I met my childhood self"
]


# Test the model
for dream in dreams:
    #print("\n🛌 **Dream:**", dream)
    print("\n🔮 **Interpretation:\n**", generate_interpretation(dream))
