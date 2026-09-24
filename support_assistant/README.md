# Module 3 - Support Assistant

What this does: a little chatbot that answers questions about Zepto's
policies, using RAG (basically: look up the relevant policy doc first,
then answer based on that instead of just making stuff up). Runs fully
offline by default using MOCK_LLM - no api key or real AI model needed
for the graded part.

## My files
- `docs/doc_01.txt` ... `doc_08.txt` - the 8 policy documents we were told to use
- `ingest.py` - loads the docs, turns them into embeddings, saves to chromadb
- `prompt_template.py` - the structured prompt (only actually used if you
  set MOCK_LLM=0, which I didn't hook up to a real model)
- `schema.py` - the pydantic models for the request/response shape
- `graph.py` - the langgraph pipeline with 3 steps: classify the question,
  then either look stuff up or just give a generic reply
- `main.py` - wraps graph.py in a fastapi app so you can call it over http
- `demo_calls.py` - quick script to test 2 example questions
- `Dockerfile` - so it can run in docker too

## How MOCK_LLM works (this confused me at first)
By default (or MOCK_LLM=1), nothing calls a real AI model anywhere -
classify_intent just checks for keywords, and the final answer is either
a canned "Based on the retrieved context: ..." string or a fixed generic
reply. This is what's actually graded. If you set MOCK_LLM=0 it would
call a real LLM instead but I left that as basically a TODO since it's
optional and not required.

## How the pieces connect (ingestion -> embedding -> retrieval -> generation)
1. ingestion: ingest.py reads the 8 txt files
2. embedding: same script turns each doc into numbers using
   sentence-transformers, saves into chroma_db/ folder
3. retrieval: graph.py's retrieve_and_answer function embeds whatever the
   user asked and asks chromadb for the 3 closest matching docs - this
   part ALWAYS runs for real, doesn't matter what MOCK_LLM is set to
4. generation: this is the ONLY part that changes based on MOCK_LLM -
   default mode just builds a canned string using the top result instead
   of asking an actual LLM to write something

## How to run it
```
pip install -r requirements.txt
python ingest.py
python demo_calls.py
uvicorn main:app --reload
```
Once uvicorn is running, go to `http://127.0.0.1:8000/docs` in a browser
to actually try it (there's no page at just `/`, only `/ask` and
`/health` exist, so hitting the bare root URL gives a 404 - that's
expected, not a bug).

### or with docker
```
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

## REAL RUN RESULTS
- `ingest.py` loaded and embedded all 8 docs fine (downloads the
  ~85MB `all-MiniLM-L6-v2` model from Hugging Face the first time,
  then it's cached locally)
- `demo_calls.py` output:
  - "What is your delivery policy?" -> correctly routed to retrieval,
    top source was `doc_01` (the actual delivery policy doc) with
    `doc_02` and `doc_05` as the other 2 matches, confidence 1.0.
    This confirms the REAL embedding model retrieves the right
    document, not just the routing logic.
  - "What's the weather like today?" -> correctly routed to the fixed
    generic reply, empty sources, confidence 1.0
- `uvicorn main:app --reload` started up fine and served requests -
  the 404s in the log are just from opening the bare root URL in a
  browser (no route defined there on purpose), not a real error
