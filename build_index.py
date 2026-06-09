import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "lol_champions"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS = 1200
OVERLAP_CHARS = 150
BATCH_SIZE = 256


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_document(text):
    name = ""
    title = ""
    preamble = []
    sections = []
    current_title = None
    current_body = []
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            header = line[2:].strip()
            name, title = header, ""
            for separator in (" \u2014 ", " - "):
                if separator in header:
                    name, _, title = header.partition(separator)
                    break
            name = name.strip()
            title = title.strip()
        elif line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        elif current_title is None:
            preamble.append(line)
        else:
            current_body.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_body).strip()))
    return name, title, "\n".join(preamble).strip(), sections


def chunk_section(body):
    body = body.strip()
    if not body:
        return []
    if len(body) <= MAX_CHARS:
        return [body]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > MAX_CHARS:
            chunks.append("\n\n".join(current))
            carry = []
            carry_len = 0
            for previous in reversed(current):
                if carry_len + len(previous) > OVERLAP_CHARS:
                    break
                carry.insert(0, previous)
                carry_len += len(previous)
            current = carry
            current_len = carry_len
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunks_for_file(path):
    text = path.read_text(encoding="utf-8")
    name, title, preamble, sections = parse_document(text)
    if not name:
        name = path.stem
    pieces = []
    if preamble.strip():
        pieces.append(("Overview", preamble.strip()))
    for section_title, body in sections:
        for piece in chunk_section(body):
            pieces.append((section_title, piece))
    records = []
    for position, (section_title, body) in enumerate(pieces):
        if not body.strip():
            continue
        label = f"{name} ({title})" if title else name
        document = f"{label} \u2014 {section_title}\n{body}"
        chunk_id = f"{slugify(name)}__{slugify(section_title)}__{position}"
        metadata = {
            "champion": name,
            "title": title,
            "section": section_title,
            "source": path.name,
        }
        records.append((chunk_id, document, metadata))
    return records


def collect_all_chunks():
    files = sorted(DATA_DIR.glob("*.txt"))
    records = []
    for path in files:
        records.extend(build_chunks_for_file(path))
    return files, records


def embed_and_store(records):
    import shutil
    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(MODEL_NAME)
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    ids = [record[0] for record in records]
    documents = [record[1] for record in records]
    metadatas = [record[2] for record in records]
    embeddings = model.encode(
        documents, batch_size=BATCH_SIZE, show_progress_bar=True
    ).tolist()

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
    return collection, model


def run_test_queries(collection, model):
    questions = [
        "what does Aatrox's ultimate do?",
        "which champion can charm enemies?",
        "tell me about Akali and the Kinkou Order",
    ]
    for question in questions:
        query_embedding = model.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        print(f"\nQ: {question}")
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            snippet = document.replace("\n", " ")
            print(f"  [{metadata['champion']} / {metadata['section']}] "
                  f"(dist {distance:.3f}) {snippet}")


def main():
    if not DATA_DIR.exists():
        print(f"Folder {DATA_DIR}/ not found. Run scraper.py first.")
        return
    files, records = collect_all_chunks()
    print(f"Found {len(files)} champion files")
    print(f"Built {len(records)} chunks")
    if not records:
        return
    collection, model = embed_and_store(records)
    print(f"Stored {collection.count()} chunks in {STORE_DIR}/")
    run_test_queries(collection, model)


if __name__ == "__main__":
    main()