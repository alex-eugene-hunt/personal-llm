import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "./fine_tuned_model"

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Create chatbot pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define a clear system prompt that tells the model to only output your personal answer.
system_prompt = (
    "You are Alex Hunt, a Software Engineer and Data Scientist based in San Francisco. "
    "Answer the following question with a concise, single-sentence response using your personal data. "
    "Do not include any additional questions or extra details in your answer.\n\n"
)

print("🚀 Personal Chatbot Ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Build the prompt by combining the system instruction with the user's question.
    prompt = system_prompt + f"Q: {user_input} A:"

    response = chatbot(
        prompt,
        max_length=150,
        do_sample=True,
        temperature=0.2,          # Lower temperature for more deterministic output
        top_p=0.85,
        repetition_penalty=1.2,   # Helps prevent loops
        truncation=True,
        return_full_text=False
    )

    # Print only the generated answer (strip extra whitespace)
    print("\n🤖:", response[0]['generated_text'].strip(), "\n")