from google import genai
import os
from dotenv import load_dotenv
import chromadb
from pathlib import Path
import pymupdf4llm
import chromadb.utils.embedding_functions as embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))
chroma_client = chromadb.PersistentClient(path="Phase_2/chroma_db")

# Document extraction

if not Path("Phase_2/Resume.md").exists():
    md_text = pymupdf4llm.to_markdown("Phase_2/Resume.pdf")

    Path("Phase_2/Resume.md").write_bytes(md_text.encode())

md_doc = Path("Phase_2/Resume.md").read_text()

# Chunking

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)

chunks = splitter.split_text(md_doc)

# Embedding

""" google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)
google_ef(["document1", "document2"])

# pass documents to query for .add and .query
collection = client.create_collection(name="name", embedding_function=google_ef)
collection = client.get_collection(name="name", embedding_function=google_ef)
 """

collection = chroma_client.get_or_create_collection(name="my_collection")

for i, chunk in enumerate(chunks, start=1):
    collection.upsert(ids=[f"document{i}"], documents=[chunk.page_content])

history = []

while True:
    user_input = input("Enter your prompt: ")

    if user_input == "exit":
        break

    results = collection.query(
        query_texts=[user_input],
        n_results=5,  # how many results to return
    )["documents"][0]

    print(results)

    prompt = f"{user_input}. Use this as context for answering: {results}"

    history.append(
        {
            "type": "user_input",
            "content": [{"type": "text", "text": prompt}],
        }
    )

    interaction = client.interactions.create(
        model="gemini-2.5-flash", store=False, input=history
    )

    print(interaction.output_text)

    for step in interaction.steps:
        history.append(step.model_dump())
