# Build Your Own Search Engine

Notebook and notes for the "Build Your Own Search Engine" workshop.

* Original workshop by Alexey Grigorev: https://github.com/alexeygrigorev/build-your-own-search-engine
* Video: https://www.youtube.com/watch?v=nMrGK5QgPVE
* All the code lives in [`notebook.ipynb`](notebook.ipynb)

What we do here:

* Use the Zoomcamp FAQ documents
    * [DE Zoomcamp](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit)
    * [ML Zoomcamp](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
    * [MLOps Zoomcamp](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit)
* Build a search engine that retrieves these documents
* The results can later feed a [Q&A RAG system](https://github.com/alexeygrigorev/llm-rag-workshop)
* [Reference implementation for text search](https://github.com/alexeygrigorev/minsearch)

## Outline

1. **Preparing the Environment**
2. **Basics of Text Search**
    - Basics of Information Retrieval
    - Vector spaces, bag of words, TF-IDF
3. **Implementing Basic Text Search**
    - TF-IDF scoring with sklearn
    - Keyword filtering with pandas
    - A class for relevance search
4. **Embeddings and Vector Search**
    - Vector embeddings
    - SVD / LSA for document embeddings
    - NMF for interpretable embeddings
    - BERT embeddings
5. **Combining Text and Vector Search**
6. **Practical Implementation Aspects and Tools**
    - Inverted indexes for text search
    - LSH for vector search (random projections)
    - Lucene/Elasticsearch, FAISS and vector databases

---

## 1. Preparing the environment

Any environment works — Codespaces, a local venv, Colab.

```bash
pip install -r requirements.txt
jupyter notebook
```

(`torch` and `transformers` in there are only needed for the BERT part of
section 4 — drop those three lines if you're skipping it.)

### Downloading the data

The FAQ documents come as a JSON file, grouped by course. We flatten it into one
flat list of records, copying the course name onto every document, so that later
we can filter by course with a single field lookup:

```python
import requests

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Each record has four fields: `course`, `section`, `question`, `text`.

### Creating the dataframe

```python
import pandas as pd

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()
```

The row number in this dataframe is the document id for the rest of the workshop —
every score array we build is aligned with it, so `np.argsort` on a score array
gives us row positions we can feed straight back into `df.iloc`.

## 2. Basics of Text Search

- **Information Retrieval** — obtaining relevant information from large datasets based on user queries.
- **Vector Spaces** — a mathematical representation where text is turned into vectors (points in space), so documents can be compared quantitatively.
- **Bag of Words** — a simple text representation treating a document as a collection of words, ignoring grammar and word order but keeping multiplicity.
- **TF-IDF (Term Frequency–Inverse Document Frequency)** — a statistical measure of how important a word is to a document within a corpus. It grows with the number of times the word appears in the document, and is offset by how often the word appears across the corpus.

### Keyword filtering

The simplest possible "search" is an exact match on a field:

```python
df[df.course == 'data-engineering-zoomcamp'].head()
```

This is a filter, not relevance: it answers "which rows equal this value", not
"which rows are most about this topic". Filtering is still useful — we combine it
with relevance scoring later, to restrict results to one course.

## 3. Implementing Basic Text Search

### Vectorization

For CountVectorizer and TF-IDF we start with a tiny example, so the matrices are
small enough to read:

```python
docs_examples = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]
```

Count vectorizer first:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(docs_examples)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs
```

This representation is the "bag of words" — we ignore word order and only keep
the words themselves. In many cases this is already good enough.

Two knobs worth knowing:

* `stop_words='english'` drops words like *on*, *not*, *for*, *after* — they occur
  everywhere and carry no signal about what a document is about.
* `min_df=5` keeps only tokens appearing in at least 5 documents. On the full FAQ
  it cuts the vocabulary from thousands of noisy tokens (typos, one-off names) to
  a manageable set.

Now the same thing with `TfidfVectorizer`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english')
X = cv.fit_transform(docs_examples)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs.round(2)
```

Same shape as the count matrix, different numbers: instead of raw counts, each
cell is weighted down if the word is common across the corpus.

### Query-Document Similarity

The query is represented in the same vector space — i.e. with the same,
already-fitted vectorizer. That is why we call `transform`, not `fit_transform`:
the query must land in the same columns as the documents.

```python
query = "Do I need to know python to sign up for the January course?"

q = cv.transform([query])
q.toarray()
```

Words in the query that are not in the fitted vocabulary are simply dropped.

We can look at the query vector and one document vector side by side:

```python
query_dict = dict(zip(names, q.toarray()[0]))
query_dict

doc_dict = dict(zip(names, X.toarray()[1]))
doc_dict
```

The more words they have in common, the better the match. Multiply the two
vectors element-wise and sum:

```python
df_qd = pd.DataFrame([query_dict, doc_dict], index=['query', 'doc']).T

(df_qd['query'] * df_qd['doc']).sum()
```

Only the terms present in *both* vectors contribute — everywhere else one of the
factors is 0. This is a dot product, so we can score every document at once with
one matrix multiplication:

```python
X.dot(q.T).toarray()
```

Watch [this linear algebra refresher](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/08-linear-algebra.md) if matrix multiplication is rusty.

Bottom line: a very fast and effective way of computing similarities.

In practice we usually use cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(X, q)
```

Cosine similarity is the dot product divided by the lengths of both vectors, so
it measures the *angle* between them and ignores document length — otherwise long
documents would win just for being long. `TfidfVectorizer` already outputs
L2-normalised vectors, so here the dot product and the cosine similarity give
identical results.

### Vectorizing all the documents

One vectorizer per field, because each field has its own vocabulary and its own
statistics:

```python
fields = ['section', 'question', 'text']
vectorizers = {}
matrices = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=5)
    X = cv.fit_transform(df[field])

    vectorizers[field] = cv
    matrices[field] = X

vectorizers['text'].get_feature_names_out()
matrices['text']
```

`section` ends up with a small vocabulary (a handful of repeated headings),
`text` with the largest one.

### Search

Search on the `text` field:

```python
query = "I just signed up. Is it too late to join the course?"

q = vectorizers['text'].transform([query])
score = cosine_similarity(matrices['text'], q).flatten()
```

Restrict to the data engineering course. Multiplying by a 0/1 mask zeroes out the
score of every document from another course, which pushes them to the bottom of
the ranking:

```python
mask = (df.course == 'data-engineering-zoomcamp').values
score = score * mask
```

Top results:

```python
import numpy as np

idx = np.argsort(-score)[:10]
```

Negating the score turns `argsort` (ascending) into a descending sort. Note:
[np.argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html)
is a more efficient way of doing the same thing — it only needs to find the top k,
not order the whole array.

Get the docs:

```python
df.iloc[idx].text
```

### Search with all the fields & boosting + filtering

We score every field and add the scores up. Boosting `question` multiplies its
contribution, because a hit in the question is usually a stronger signal of
relevance than a hit somewhere in the middle of a long answer:

```python
boost = {'question': 3.0}

score = np.zeros(len(df))

for f in fields:
    b = boost.get(f, 1.0)
    q = vectorizers[f].transform([query])
    s = cosine_similarity(matrices[f], q).flatten()
    score = score + b * s
```

And the filters:

```python
filters = {
    'course': 'data-engineering-zoomcamp'
}

for field, value in filters.items():
    mask = (df[field] == value).values
    score = score * mask
```

Results:

```python
idx = np.argsort(-score)[:10]
results = df.iloc[idx]
results.to_dict(orient='records')
```

### Putting it all together

Same code, wrapped in a class:

```python
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
```

Using it:

```python
index = TextSearch(
    text_fields=['section', 'question', 'text']
)
index.fit(documents)

index.search(
    query='I just signed up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)
```

The same implementation, packaged: https://github.com/alexeygrigorev/minsearch

**Note**: this is a toy example illustrating how relevance search works. It is
not meant for production — it scores every document on every query.

## 4. Embeddings and Vector Search

The problem with keyword matching: it only finds exact matches. A query saying
"sign up" will not match a document saying "register", even though they mean the
same thing.

### What are Embeddings?

- **Conversion to Numbers:** embeddings turn words, sentences and documents into dense vectors (arrays of numbers).
- **Capturing Similarity:** similar items get similar vectors, so closeness in the vector space means closeness in meaning.
- **Dimensionality Reduction:** embeddings compress complex characteristics into a few hundred (or a few dozen) numbers.
- **Use in Machine Learning:** these vectors feed models for recommendations, text analysis and pattern recognition.

Dense vs sparse: a TF-IDF row has one column per vocabulary word and is almost
all zeros. An embedding has a fixed small number of columns, nearly all non-zero,
each one a learned combination of many words.

### SVD

Singular Value Decomposition is the simplest way to turn a bag-of-words
representation into embeddings. Applied to a term-document matrix it is also
known as LSA — Latent Semantic Analysis.

We still don't preserve word order (it wasn't in the bag-of-words to begin with),
but we reduce dimensionality and capture synonyms: two words that keep appearing
in the same documents end up pulled onto the same component.

We won't go into the mathematics — it is enough to know that SVD "compresses" the
input vectors so that as much of the original information as possible is retained.

The compression is lossy — we cannot restore 100% of the original vector, but the
result is close enough.

Example with images:

<img src="http://habrastorage.org/files/855/a65/c62/855a65c624dc4174b526fb5e03b98555.png" />

Let's take the vectorizer for the `text` field and turn its matrix into embeddings:

```python
from sklearn.decomposition import TruncatedSVD

X = matrices['text']
cv = vectorizers['text']

svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)

X_emb[0]
```

`n_components=16` is the size of the embedding: every document goes from a
vocabulary-sized sparse row to 16 numbers.

The query goes through exactly the same two steps — vectorizer, then SVD — so
that it lands in the same 16-dimensional space:

```python
query = 'I just signed up. Is it too late to join the course?'

Q = cv.transform([query])
Q_emb = svd.transform(Q)
Q_emb[0]
```

Similarity between the query and a document:

```python
np.dot(X_emb[0], Q_emb[0])
```

And for all documents — same code as before, just on dense embeddings instead of
the sparse matrix:

```python
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
```

### Non-Negative Matrix Factorization

SVD produces negative numbers, which are hard to interpret.

NMF (Non-Negative Matrix Factorization) is a similar idea, except that for
non-negative input matrices it produces non-negative results.

We can read each column (feature) of the embedding as a topic/concept, and the
value as how much the document is about that concept.

For the documents:

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=16)
X_emb = nmf.fit_transform(X)
X_emb[0]
```

And for the query:

```python
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]
```

Similarity computed the same way as before:

```python
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
```

### BERT

The problem with the two previous approaches is that they ignore word order. They
treat every word separately — that's why it's called "bag of words".

BERT and other transformer models don't have this problem: the embedding of a
word depends on the words around it.

Install:

```bash
pip install transformers tqdm torch
```

Load the model:

```python
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode if not training
```

We need:

- tokenizer — for turning text into token ids
- model — for compressing the text into embeddings

First we tokenize:

```python
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

`padding=True` makes all sequences in a batch the same length; `truncation=True`
cuts anything longer than the model's maximum input length.

Then we compute the embeddings:

```python
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state
```

`hidden_states` has shape (sentences, tokens, 768) — one vector *per token*. We
need one vector per sentence, so we average over the token axis:

```python
sentence_embeddings = hidden_states.mean(dim=1)
sentence_embeddings.shape
```

And convert to a numpy array:

```python
X_emb = sentence_embeddings.numpy()
```

If you use a GPU, first move the tensors to CPU:

```python
sentence_embeddings_cpu = sentence_embeddings.cpu()
```

Now for all our texts. BERT is slow and memory-hungry, so we go in batches:

```python
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result
```

And use it:

```python
from tqdm.auto import tqdm
texts = df['text'].tolist()
text_batches = make_batches(texts, 8)

all_embeddings = []

for batch in tqdm(text_batches):
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state

        batch_embeddings = hidden_states.mean(dim=1)
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)

final_embeddings = np.vstack(all_embeddings)
```

As a function:

```python
def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, batch_size)

    all_embeddings = []

    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state

            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)

    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings
```

And use it:

```python
embeddings = {}

for f in fields:
    print(f'computing embeddings for {f}...')
    embeddings[f] = compute_embeddings(df[f].tolist())
```

## 5. Combining Text and Vector Search

The two methods fail in opposite ways:

* keyword/TF-IDF search is precise on exact words and blind to synonyms
* vector search catches synonyms and paraphrases, but happily returns something
  vaguely on-topic that misses the exact term you typed

So run both and merge. The simplest merge is a weighted sum of the two scores —
one number, `alpha`, decides how much you trust each side:

```python
query = 'I just signed up. Is it too late to join the course?'

q = vectorizers['text'].transform([query])          # sparse, vocabulary space
q_svd = svd.transform(q)                             # dense, 16-dim space
X_svd = svd.transform(matrices['text'])

text_score = cosine_similarity(matrices['text'], q).flatten()
vector_score = cosine_similarity(X_svd, q_svd).flatten()

alpha = 0.5
score = alpha * text_score + (1 - alpha) * vector_score

idx = np.argsort(-score)[:5]
list(df.iloc[idx].text)
```

`alpha = 1.0` is pure keyword search, `alpha = 0.0` is pure vector search.

One caveat for the weighted sum: it only makes sense if both scores are on the
same scale. Here they are — both are cosine similarities — but a raw BM25 score
and a cosine similarity are not comparable, and adding them lets whichever has
the bigger numbers dominate.

The scale-free alternative is to combine *ranks* instead of scores. Reciprocal
Rank Fusion (RRF) gives each document `1 / (k + rank)` from each ranking and adds
those up:

```python
from collections import defaultdict

def rrf(*rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

text_rank = np.argsort(-text_score)[:10]
vector_rank = np.argsort(-vector_score)[:10]

idx = rrf(text_rank, vector_rank)[:5]
list(df.iloc[idx].text)
```

Documents that both methods like float to the top; documents only one method
found still get in, just lower. The constant `k` damps the influence of the very
top positions — it is a knob, and 60 is the value from the original paper.

## 6. Practical Implementation Aspects and Tools

Everything above scores *every* document for *every* query. With 1000 FAQ entries
that's instant. With 100 million documents it is hopeless. Real systems avoid the
full scan on both sides: an inverted index for text, and an approximate nearest
neighbour structure for vectors.

### Inverted indexes for text search

A forward index maps document → words. Invert it: word → the documents containing
it. Then a query only has to look at documents that share at least one term with
it, instead of all of them.

```python
import re
from collections import defaultdict

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

inverted_index = defaultdict(set)

for doc_id, text in enumerate(df.text):
    for token in tokenize(text):
        inverted_index[token].add(doc_id)

len(inverted_index)     # vocabulary size
```

Look-ups are set operations:

```python
inverted_index['docker'] & inverted_index['kafka']   # both words (AND)
inverted_index['docker'] | inverted_index['kafka']   # either word (OR)
```

Used as a first stage, it shrinks how much we have to score:

```python
def candidates(query):
    ids = set()
    for token in tokenize(query):
        ids |= inverted_index.get(token, set())
    return sorted(ids)

c = candidates('how do I run kafka in docker')
len(c), len(df)     # score this many documents instead of all of them
```

The full scoring formula (TF-IDF, or BM25 in most real engines) then runs only on
the candidates. The index also stores the term frequencies alongside each doc id,
so scoring needs no second pass over the text.

### LSH for vector search (random projections)

Vectors have no words to index, so the trick is different: hash nearby vectors
into the same bucket *on purpose*. Locality-Sensitive Hashing does exactly the
opposite of a cryptographic hash — similar inputs should collide.

The simplest LSH for cosine similarity is random projections: pick a few random
hyperplanes, and record which side of each plane the vector falls on. That's one
bit per plane, and vectors pointing in a similar direction get the same bits.

```python
np.random.seed(1)

X_svd = svd.transform(matrices['text'])
n_bits = 6
planes = np.random.randn(X_svd.shape[1], n_bits)

def lsh_hash(V):
    bits = (V @ planes) > 0
    return [''.join('1' if b else '0' for b in row) for row in bits]

buckets = defaultdict(list)
for doc_id, h in enumerate(lsh_hash(X_svd)):
    buckets[h].append(doc_id)

len(buckets)    # how many buckets got used
```

At query time we hash the query and only score its bucket:

```python
q_svd = svd.transform(vectorizers['text'].transform([query]))
bucket = buckets.get(lsh_hash(q_svd)[0], [])

len(bucket), len(df)   # if this comes back empty, lower n_bits

score = cosine_similarity(X_svd[bucket], q_svd).flatten()
idx = np.array(bucket)[np.argsort(-score)[:5]]
list(df.iloc[idx].text)
```

This is *approximate*: a document just on the other side of one hyperplane lands
in a different bucket and is never seen, so we trade recall for speed. Real
implementations build several independent hash tables and search the union of the
matching buckets, which makes a miss in one table survivable. More bits = smaller
buckets = faster but less accurate; fewer bits = the opposite.

HNSW, the graph-based index used by most current vector databases, solves the same
problem in a different way, but the trade-off is identical: approximate results in
exchange for not scanning everything.

### Technologies

Text search:

* **Lucene** — the Java library implementing the inverted index and BM25 scoring.
* **Elasticsearch** / **OpenSearch** — distributed search engines built on Lucene, adding an HTTP API, sharding and replication. This is what you reach for instead of the `TextSearch` class above.

Vector search:

* **FAISS** — Facebook AI Similarity Search, a library of ANN indexes (flat, IVF, HNSW, product quantization). A library, not a server: it lives in your process.
* **Vector databases** — Qdrant, Weaviate, Milvus, Chroma, Pinecone: they add persistence, metadata filtering and an API on top of an ANN index.
* **pgvector** — a Postgres extension, if you'd rather not run another service.

And Elasticsearch/OpenSearch, Qdrant and the rest now also support hybrid search —
running the keyword and vector query together and fusing the results, usually with
RRF, which is section 5 done for you.

## 7. Conclusion — the road from keyword filtering to a fine-tuned LLM

Every step in this notebook exists because the step before it failed at something
specific. Read the diagram top to bottom: the left column is the idea, the arrow
label is the problem that forced the next idea, the right branches are the
production tools that shipped each idea.

```
  ══════════ 1. EXACT WORDS ═══════════════════════════════════════════════

   keyword filtering  (boolean retrieval, 1960s)
          │  matches or doesn't - no way to rank 900 hits
          ▼
   TF-IDF  (Sparck Jones, 1972)
          │  now we have a weight per word, but no way to compare two docs
          ▼
   cosine similarity / vector space model  (Salton, 1975)
          │  long documents still game the score
          ▼
   BM25  (Robertson & Walker, 1994) ────► Lucene 1999 ──► Elasticsearch 2010
          │                                                       │
          │                                                       ▼
          │                                                  OpenSearch 2021
          │  "sign up" still never matches "register"
          ▼
  ══════════ 2. MEANING, BAG OF WORDS ═════════════════════════════════════

   LSA / SVD  (Deerwester et al., 1990)
          │  16 dense numbers instead of 1333 sparse ones - synonyms collapse
          │  together. But the numbers go negative and mean nothing to a human
          ▼
   NMF  (Lee & Seung, 1999)
          │  non-negative, so each column reads as a topic
          │  still no word order: "docker in kafka" == "kafka in docker"
          ▼
  ══════════ 3. MEANING, WORD ORDER ═══════════════════════════════════════

   word2vec / GloVe  (2013 / 2014)
          │  vectors learned from context, not counts - but one vector per word,
          │  so "bank" has a single meaning
          ▼
   Transformer  (Vaswani et al., 2017)
          │  attention: every token is read in the context of the others
          ▼
   BERT  (Devlin et al., 2018)
          │  contextual embeddings - but mean-pooled BERT (what we did above)
          │  was never trained to make similar sentences land close together
          ▼
   Sentence-BERT 2019 / DPR 2020 ──► LSH 1998 ──► HNSW 2016, FAISS 2017
          │   embeddings trained                          │
          │   for retrieval                               ▼
          │                                    vector DBs: Qdrant, Weaviate,
          │  vectors miss the exact term       Milvus, Chroma, pgvector
          │  you actually typed
          ▼
  ══════════ 4. BOTH AT ONCE ══════════════════════════════════════════════

   hybrid search: BM25 + dense vectors, fused with RRF  (Cormack et al., 2009)
          │  we now return excellent documents. The user wanted an answer
          ▼
  ══════════ 5. ANSWERS, NOT LINKS ════════════════════════════════════════

   RAG  (Lewis et al., 2020)  =  this search engine  +  an LLM
          │  retrieval grounds the model in your documents instead of its weights
          ▼
   LLM  (GPT-3, 2020)  ──►  instruction tuning / RLHF  (InstructGPT, 2022)
          │  generic model, generic tone, no knowledge of your domain jargon
          ▼
   fine-tuning  (full FT, or LoRA / PEFT, Hu et al., 2021)
             teach format, tone and domain vocabulary - NOT facts:
             facts stay in the retrieval layer, because they change
```

### Why each step exists

| # | Step | The problem it solves | Why it could not come earlier |
|---|------|----------------------|-------------------------------|
| 1 | Keyword filtering | Find documents containing a term at all | The baseline |
| 2 | TF-IDF | 900 documents match — which one first? Rare words carry more signal than common ones | Needs a corpus to compute document frequency |
| 3 | Cosine similarity | Turn two weight vectors into one comparable number, independent of length | Needs vectors, i.e. the vector space model |
| 4 | BM25 | The 20th "kafka" in a doc is not worth as much as the 2nd; long docs shouldn't win by default | A probabilistic refinement of TF-IDF |
| 5 | Inverted index → Lucene → Elasticsearch/OpenSearch | Scoring all N documents per query doesn't scale | Engineering, not new maths: needed the scoring formula to be settled first |
| 6 | SVD / LSA | Synonyms. "sign up" and "register" are different columns and can never match | Needs the term-document matrix from steps 2–3 |
| 7 | NMF | SVD components go negative and are uninterpretable | Same input, a constraint added |
| 8 | word2vec / GloVe | LSA meaning comes from co-occurrence counts in one corpus; learn it from context on billions of words instead | Needed cheap GPUs and large text dumps |
| 9 | Transformer | Bag of words and static word vectors both throw away order and context | Needed attention (2014) plus the hardware to train it |
| 10 | BERT | One vector per word can't disambiguate; BERT gives one vector *per occurrence* | Built directly on the Transformer encoder |
| 11 | Sentence-BERT / DPR | Mean-pooled BERT is a weak similarity metric — it was trained to fill in masked words, not to rank | Needs labelled query–document pairs to fine-tune on |
| 12 | LSH / HNSW / FAISS / vector DBs | Comparing a query to 100M dense vectors is a full scan | Only becomes a problem once dense retrieval actually works |
| 13 | Hybrid search + RRF | Lexical misses synonyms, dense misses exact terms, IDs and error codes | Needs both retrievers to exist |
| 14 | RAG | The LLM's knowledge is frozen at training time and it invents citations | Needs a retriever *and* a model good enough to read the passages |
| 15 | LLM + instruction tuning | Users want an answer, not ten links | Needs the scale of GPT-3 and then RLHF to follow instructions |
| 16 | Fine-tuning / LoRA | Format, tone, domain vocabulary the base model gets wrong | Last resort: it's the expensive knob, and retrieval already fixed the facts |

### The practical takeaway

The order above is also the order to **build** in. Steps 1–5 (BM25 in Elasticsearch)
solve most of a real search problem for a fraction of the cost. Add dense retrieval
when you measure that synonyms are actually hurting recall, hybrid when you measure
that dense alone drops exact matches, and fine-tune last — after retrieval, prompting
and the ranking are all exhausted.

This notebook stopped at step 13. Step 14 onwards is the
[LLM RAG workshop](https://github.com/alexeygrigorev/llm-rag-workshop).
