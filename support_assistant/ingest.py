# ingest.py
# Step 1 - read all the policy docs, turn them into embeddings (basically
# turning text into a list of numbers that captures what it "means"), and
# save those into a chromadb database so I can search them later.

import glob
import chromadb
from sentence_transformers import SentenceTransformer

# I'm treating each whole doc file as ONE chunk since they're already
# short (like one paragraph each), didn't feel like i needed to split
# them up any smaller than that

doc_ids = []
doc_texts = []

# grab every file that matches docs/doc_something.txt
file_list = sorted(glob.glob("docs/doc_*.txt"))

for file_path in file_list:
    # this pulls out just "doc_01" from "docs/doc_01.txt"
    just_the_name = file_path.split("/")[-1]
    doc_id = just_the_name.replace(".txt", "")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    doc_ids.append(doc_id)
    doc_texts.append(text)

print("loaded", len(doc_texts), "docs:", doc_ids)

# now load the embedding model - this is free and runs locally, no api key
print("loading the embedding model, this might take a sec the first time...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# turn all our doc texts into embeddings (a list of numbers per doc)
embeddings = model.encode(doc_texts)
embeddings = embeddings.tolist()  # chromadb wants plain python lists not numpy arrays

# connect to (or create) our local chromadb folder
client = chromadb.PersistentClient(path="chroma_db")

# using get_or_create + upsert here instead of create + add, so that if I
# run this script again later it doesn't crash complaining about duplicate
# ids, it just overwrites them
collection = client.get_or_create_collection("zepto_policies")
collection.upsert(ids=doc_ids, embeddings=embeddings, documents=doc_texts)

print("saved", len(doc_ids), "chunks into the chroma_db folder, done!")
