# RAG PDF Assistant

Ask questions about a PDF in plain English and get answers that actually come from
the document — with the page numbers shown so you can check.

I built this to understand how retrieval-augmented generation works underneath,
rather than just wiring a tutorial together. The notebook shows every step.

---

## What it does

The demo document is a 12-page employee handbook. Ask it anything about that
handbook and it answers from the text. Ask it something the handbook doesn't cover
and it tells you it doesn't know.

That second part is harder than it sounds. Vector search always returns results —
it hands back the closest matches whether or not any of them are relevant. Ask
about dental insurance and you'll still get three chunks about leave policy and
office safety. The system only refuses because the prompt tells it to.

| Question | Answer |
|---|---|
| How much parental leave do employees get? | 26 weeks primary, 8 weeks secondary — from page 4 |
| What is the dental insurance policy? | "The information is not available in the provided document." |

---

## How it's built

| Piece | What I used |
|---|---|
| Reading the PDF | PyPDFLoader |
| Splitting it up | RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2, runs locally, 384 dimensions |
| Vector search | FAISS |
| The model | gpt-oss-20b on Groq, temperature 0 |
| Glue | LangChain |
| Web app | Flask and plain JavaScript |

The flow:

```
PDF → 12 pages → 35 chunks → vectors → FAISS index (saved to disk)

question → vector → 3 nearest chunks → prompt → answer + sources
```

Indexing happens once and gets written to disk. The web app loads that file at
startup instead of rebuilding, which is the difference between a one-second
response and a thirty-second one.

---

## Why chunk size is 800

This is the part I'd point at first.

Most guides tell you to pick a chunk size and move on. I built three indexes at
2000, 800 and 400 characters and ran the same questions through each. FAISS
returns distance, so lower is better.

| Question | 2000 | 800 | 400 |
|---|---|---|---|
| Parental leave | 0.83 | 0.76 | 0.70 |
| Firmware laptop | 0.99 | 0.82 | 0.70 |
| Dental insurance (not in the document) | 1.56 | 1.55 | 1.46 |
| **Gap between the two** | 0.73 | **0.79** | 0.76 |

Smaller chunks matched real questions better. The interesting bit is the last row:
questions with no answer stayed far away, so the gap between "found it" and
"nothing here" got wider rather than blurrier. That's not guaranteed — it was worth
checking rather than assuming.

At 2000 characters the top three results for a leave question included an unrelated
chunk about working hours. At 800, all three came from the right page.

I went with 800 over 400 even though 400 scored slightly better, because 400-character
chunks cut sentences in half. The model has to read these — coherence matters more
than the last 0.06.

Rough rule for this index: under 1.0 is a real match, 1.0 to 1.3 is weak but usually
still relevant, over 1.4 means there's probably nothing there.

---

## Running it yourself

```bash
git clone https://github.com/prasanth1218/rag-pdf-assistant.git
cd rag-pdf-assistant

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac or Linux

pip install -r requirements.txt
```

Grab a free key from https://console.groq.com/keys and put it in a `.env` file:

```
GROQ_API_KEY=gsk_your_key_here
```

Then:

```bash
python ingest.py     # builds the index, takes a few seconds
python app.py        # starts the server
```

Open http://127.0.0.1:5000

---

## About deployment

It runs fine locally but won't stay up on a 512MB free tier. PyTorch holds 300-400MB
in memory just from being imported, before the embedding model even loads. The build
succeeds, the app boots, and then it gets killed.

Switching to CPU-only PyTorch cut the install from about 2.9GB to 200MB — the default
Linux wheel drags in 2.5GB of CUDA libraries for a machine with no GPU — but that's
download size, not runtime memory. Running the embeddings through ONNX instead of
PyTorch would probably fit. I haven't done that yet.

---

## Swapping in your own PDF

Drop it in `documents/`, change `PDF_PATH` at the top of `ingest.py`, and run
`python ingest.py` again.

Worth knowing: 800 characters suits a handbook full of short policy paragraphs. A
dense paper or a contract might want something different. The comparison code is in
the notebook if you want to check.

---

## What's in the repo

```
rag_project.ipynb    the whole thing built step by step, with real outputs
ingest.py            builds the index
app.py               Flask server
templates/           the page
static/              CSS
documents/           the PDF
faiss_index/         the built index
```

The notebook is the most useful file here. It has every stage with the actual numbers
I got, including the ones that surprised me.
