import ollama

def ask_local_bot(prompt):
    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides diabetes-safe lifestyle advice based on NHS guidance."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

# Simple chat loop
print("🧠 Diabetes Chatbot (Local Model) – type 'exit' to quit")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    answer = ask_local_bot(user_input)
    print("Bot:", answer)
