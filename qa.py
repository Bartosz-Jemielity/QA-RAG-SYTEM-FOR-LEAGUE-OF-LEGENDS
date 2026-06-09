import os
import sys

from dotenv import load_dotenv

from build_index import MODEL_NAME, STORE_DIR, COLLECTION_NAME

load_dotenv()

LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"
LLM_API_KEY_ENV = "OPENROUTER_API_KEY"
TOP_K = 8
CHAMPION_K = 14
MAX_DISTANCE = 0.9

SYSTEM_PROMPT = (
    "You are a League of Legends assistant. Answer the user's question using only "
    "the provided context about champions, their abilities and lore. Synthesize a "
    "complete, self-contained answer from the relevant context; describe what the "
    "context actually says instead of telling the user to read a section. If the "
    "context does not contain the answer, say you do not have that information. Be "
    "concise and mention the champion and section names you relied on."
)


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def load_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(STORE_DIR))
    return client.get_collection(COLLECTION_NAME)


def champion_names(collection):
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({metadata["champion"] for metadata in metadatas}, key=len, reverse=True)


def detect_champion(question, names):
    lowered = question.lower()
    for name in names:
        if name.lower() in lowered:
            return name
    return None


def retrieve(collection, model, question, champion=None):
    query_embedding = model.encode([question]).tolist()
    where = {"champion": champion} if champion else None
    n_results = CHAMPION_K if champion else TOP_K
    results = collection.query(
        query_embeddings=query_embedding, n_results=n_results, where=where
    )
    hits = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance <= MAX_DISTANCE:
            hits.append((document, metadata, distance))
    return hits


def build_context(hits):
    blocks = []
    for position, (document, metadata, _distance) in enumerate(hits, start=1):
        label = f"{metadata['champion']} / {metadata['section']}"
        blocks.append(f"[{position}] ({label})\n{document}")
    return "\n\n".join(blocks)


def build_messages(question, context):
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_llm(messages):
    from openai import OpenAI

    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set the {LLM_API_KEY_ENV} environment variable with your API key."
        )
    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key, timeout=30)
    response = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()


def answer_plain(question):
    messages = [
        {"role": "system", "content": "You are a League of Legends assistant."},
        {"role": "user", "content": question},
    ]
    return call_llm(messages)


def answer(collection, model, question, champion=None):
    if champion is None:
        champion = detect_champion(question, champion_names(collection))
    hits = retrieve(collection, model, question, champion)
    if not hits:
        return "No relevant context found in the knowledge base.", []
    context = build_context(hits)
    messages = build_messages(question, context)
    reply = call_llm(messages)
    sources = [
        f"{metadata['champion']} / {metadata['section']} (dist {distance:.3f})"
        for _document, metadata, distance in hits
    ]
    return reply, sources


def main():
    question = " ".join(sys.argv[1:]).strip()
    model = load_model()
    collection = load_collection()
    print(f"Loaded collection '{COLLECTION_NAME}' with {collection.count()} chunks")

    if question:
        reply, sources = answer(collection, model, question)
        print(f"\n{reply}\n")
        if sources:
            print("Sources:")
            for source in sources:
                print(f"  - {source}")
        return

    print("Ask a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        reply, sources = answer(collection, model, question)
        print(f"\n{reply}\n")
        if sources:
            print("Sources:")
            for source in sources:
                print(f"  - {source}")


if __name__ == "__main__":
    main()
