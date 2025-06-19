from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and chunk all .txt files from 'data/' with source tracking
chunk_size = 500
chunks = []
sources = []

print("🔍 Checking files in 'data/':")
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        path = os.path.join("data", filename)
        size = os.path.getsize(path)
        print(f" - {filename} ({size} bytes)")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])
                sources.append(filename)

if len(chunks) == 0:
    print("❌ No chunks created. Please check that your .txt files are not empty.")
    exit()

print(f"✅ Loaded and split all documents into {len(chunks)} chunks.")

# Step 2: Embed chunks
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Step 3: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ Embeddings created and indexed.")

# Step 4: Safety checker
def get_rag_safety_level(answer, source_chunks):
    context_text = " ".join(source_chunks)
    docs = [context_text, answer]
    vectorizer = TfidfVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()
    sim_score = cosine_similarity([vectors[1]], [vectors[0]])[0][0]
    if sim_score > 0.5:
        return "🟢 GREEN – Safe"
    elif sim_score > 0.2:
        return "🟠 AMBER – Possibly correct but unverified"
    else:
        return "🔴 RED – Unsupported or risky"

# Step 5: Chatbot function
def ask_rag_bot(question):
    q_embedding = embedder.encode([question])[0]
    D, I = index.search(np.array([q_embedding]), k=3)

    top_chunks = [chunks[i] for i in I[0]]
    top_sources = [sources[i] for i in I[0]]
    context = "\n\n".join(top_chunks)

    print("\n📥 Retrieved NHS Chunks:")
    for i, idx in enumerate(I[0]):
        print(f"\n--- Chunk {i+1} from: {sources[idx]} ---")
        print(chunks[idx][:300] + "...")

    print("\n🧠 Sending to Mistral:")
    print(f"System Prompt:\n---\n{context[:500]}...\n---")

    response = ollama.chat(
        model='mistral',
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful NHS-based assistant. Use the following NHS guidance to answer safely:\n\n{context}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print("\n💬 Bot Reply:")
    print(response['message']['content'])

    return response['message']['content'], top_chunks, top_sources

# Step 6: Logging
def log_interaction(question, answer, rating, chunks_used, chunk_sources, log_file="chat_log.csv"):
    with open(log_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            rating,
            " | ".join(f"{chunk_sources[i]}: {chunks_used[i][:150].replace('\n', ' ')}" for i in range(len(chunks_used)))
        ])

# Step 7: Chat loop
print("📚 RAG Diabetes Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    answer, source_chunks, source_files = ask_rag_bot(user_input)
    print("\nBot:", answer)

    rating = get_rag_safety_level(answer, source_chunks)
    print("📊 Safety Rating:", rating)

    log_interaction(user_input, answer, rating, source_chunks, source_files)
