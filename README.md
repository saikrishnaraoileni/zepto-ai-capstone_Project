# Zepto Data & AI Platform - Capstone Project

One repo, 3 modules, submitted together as required by the assignment.

| Module | Folder | Marks | What it does |
|---|---|---|---|
| 1 - Data Pipeline | `/data_pipeline` | 25 | Scrape books.toscrape.com, clean it, convert to INR, load into SQLite, run SQL queries |
| 2 - Analytics Pipeline | `/analytics` | 50 | EDA + ML modeling on the Titanic dataset |
| 3 - Support Assistant | `/support_assistant` | 25 | RAG chatbot answering Zepto policy questions |

## Setup

One consolidated `requirements.txt` at the repo root covers everything.

```
py -m venv venv
venv\Scripts\activate      (windows)
py -m pip install -r requirements.txt
```

(on windows, use `py` not `python` - "python" gets hijacked by a Windows
Store shortcut unless you've set it up differently)

## How to run each module

### Module 1 - Data Pipeline
```
cd data_pipeline
py run_all.py
```
Runs scrape -> clean -> build_db -> queries in order. Only `scrape.py`
needs internet.

### Module 2 - Analytics Pipeline
```
cd analytics
py 01_eda.py
py 02_modeling.py
```
`01_eda.py` needs internet the first time (seaborn downloads + caches the
titanic dataset). Both scripts save chart images into `analytics/charts/`.

### Module 3 - Support Assistant
```
cd support_assistant
py ingest.py
py demo_calls.py
py -m uvicorn main:app --reload
```
Then visit `http://127.0.0.1:8000/docs` to try it interactively (there's
no page at just `/`, only `/ask` and `/health`, so hitting the bare root
URL 404s - that's expected). Or with docker:
`docker build -t zepto-support .` then `docker run -p 7860:7860 zepto-support`.

## Design decisions + real results (full reasoning in each module's own README)

**Module 1:** fixed conversion rate 1 GBP = 105.50 INR. Rows that fail to
parse get dropped, not guessed at. Real run: scraped 107 books across 5
categories (Travel 11, Mystery 32, Historical Fiction 26, Classics 19,
Poetry 19), 0 dropped, all 5 SQL queries + the pandas/SQL join check
passed.

**Module 2:** missing-value rule is <5%=drop rows, 5-30%=impute
median/mode, >30%=drop column or make "Missing" its own category (used
for `deck`, which was 77% missing). `02_modeling.py` deliberately does
NOT use the already-cleaned data from `01_eda.py` - it re-cleans inside a
pipeline fit only on the training split, to avoid leaking test-set info
into training. Real run: 891 rows loaded, 889 after cleaning; survival
rate 74% for women vs 19% for men; best classifier by F1 was Logistic
Regression (f1=0.724, auc=0.843); SMOTE won the imbalance comparison
(f1=0.737); best Random Forest params from GridSearchCV were max_depth=8,
n_estimators=200 with an OOB score of 0.822; the fare-regression task
showed clear heteroscedasticity.

**Module 3:** RAG assistant, graded baseline runs fully offline via
MOCK_LLM (no LLM API needed). Retrieval (embeddings + ChromaDB) always
runs for real in both modes - only the final answer text is mocked. Real
run: the delivery-policy question correctly retrieved `doc_01` (the
actual delivery doc) as its top match; the unrelated weather question
correctly fell back to the generic reply with no sources.

