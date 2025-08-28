{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "613d802b-b3c3-4310-84ff-8183e7673ed6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -------------------------------------------------\n",
    "# Minimal TextRank pipeline (beginner-friendly)\n",
    "# 0) sentence_split\n",
    "# 1) build_sim_matrix\n",
    "# 2) rank_sentences\n",
    "# 3) summarize\n",
    "# -------------------------------------------------\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "5a9db43f-e662-4ead-befa-74b779e9a61c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import math\n",
    "import numpy as np\n",
    "import networkx as nx\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "d0c6ac30-525d-4c5b-bf0d-a8df7334c80b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Hello.', 'Today I am experimenting with TextRank.', 'Does it work well?']\n"
     ]
    }
   ],
   "source": [
    "from nltk.tokenize import sent_tokenize\n",
    "text = \"Hello. Today I am experimenting with TextRank. Does it work well?\"\n",
    "print(sent_tokenize(text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "92bdf904-b391-4bd0-a3d1-f111f7997ae9",
   "metadata": {},
   "outputs": [],
   "source": [
    "demo_text = (\n",
    "    \"TextRank is a graph-based ranking algorithm for NLP. \"\n",
    "    \"It builds a graph of sentences using pairwise similarity. \"\n",
    "    \"Then it applies PageRank to score sentences by importance. \"\n",
    "    \"This method is often used for extractive summarization. \"\n",
    "    \"Even beginners can implement a minimal version quickly.\"\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "04e42f2a-e630-40be-9461-a008d5471db9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📌 Original text:\n",
      "TextRank is a graph-based ranking algorithm for NLP. It builds a graph of sentences using pairwise similarity. Then it applies PageRank to score sentences by importance. This method is often used for extractive summarization. Even beginners can implement a minimal version quickly.\n"
     ]
    }
   ],
   "source": [
    "print(\"📌 Original text:\")\n",
    "print(demo_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "d0e3d201-104f-431a-9468-be607fda5d6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# 0) Sentence splitter\n",
    "#    - Prefer NLTK's sent_tokenize (more accurate)\n",
    "#    - Fallback to a simple regex if NLTK/punkt is unavailable\n",
    "# -----------------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "af426f3c-2b8c-4a40-acb3-45007c60eb84",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "def sentence_split(text):\n",
    "    \"\"\"\n",
    "    Split a long text into sentences.\n",
    "\n",
    "    Steps:\n",
    "      1) Clean whitespace (collapse multiple spaces/newlines, strip leading/trailing spaces)\n",
    "      2) Try NLTK's sent_tokenize (more accurate sentence boundary detection)\n",
    "         - If the 'punkt' model is missing, download it automatically\n",
    "      3) If NLTK is unavailable or fails, fall back to a simple regex-based splitter\n",
    "      4) Strip each sentence and remove empty strings\n",
    "    \"\"\"\n",
    "    # Step 1: Basic cleaning\n",
    "    text = re.sub(r\"\\s+\", \" \", text).strip()\n",
    "    if not text:\n",
    "        return []\n",
    "\n",
    "    try:\n",
    "        # Step 2: Attempt NLTK-based sentence tokenization\n",
    "        import nltk\n",
    "        from nltk.tokenize import sent_tokenize\n",
    "\n",
    "        # Ensure 'punkt' tokenizer data exists\n",
    "        try:\n",
    "            nltk.data.find(\"tokenizers/punkt\")\n",
    "        except LookupError:\n",
    "            nltk.download(\"punkt\", quiet=True)\n",
    "\n",
    "        sents = sent_tokenize(text)\n",
    "\n",
    "    except Exception:\n",
    "        # Step 3: Fallback to regex-based splitting\n",
    "        # Regex explanation:\n",
    "        #   (?<=[.!?]) : position immediately AFTER a period, question mark, or exclamation mark\n",
    "        #   \\s+        : one or more spaces\n",
    "        sents = re.split(r\"(?<=[.!?])\\s+\", text)\n",
    "\n",
    "    # Step 4: Final cleanup → strip whitespace and discard empties\n",
    "    return [s.strip() for s in sents if s and s.strip()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "93276f03-916d-461b-b8d5-d3dcfa6fa4ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Step 0 - Sentences:\n",
      "1. TextRank is a graph-based ranking algorithm for NLP.\n",
      "2. It builds a graph of sentences using pairwise similarity.\n",
      "3. Then it applies PageRank to score sentences by importance.\n",
      "4. This method is often used for extractive summarization.\n",
      "5. Even beginners can implement a minimal version quickly.\n"
     ]
    }
   ],
   "source": [
    "# Step 0) Sentence split\n",
    "sents = sentence_split(demo_text)\n",
    "print(\"\\n✅ Step 0 - Sentences:\")\n",
    "for i, s in enumerate(sents, 1):\n",
    "    print(f\"{i}. {s}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7af1e2a8-c9d9-424e-95b2-f3eabe2b9d8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# 1) Similarity matrix (TF-IDF + cosine)\n",
    "#    * This is where we tune \"vector settings\"\n",
    "# -----------------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "f3e1be0c-2d29-45a0-9536-ade0e8d9fc53",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_sim_matrix(sent_list):\n",
    "    \"\"\"\n",
    "    Build a sentence-by-sentence cosine similarity matrix using TF-IDF.\n",
    "    Key vector settings:\n",
    "      - stop_words=None  : good for Korean or mixed text\n",
    "                           (for pure English, you can set 'english')\n",
    "      - ngram_range=(1,2): unigrams + bigrams (helps with short sentences)\n",
    "      - lowercase=True   : normalize casing (mostly for English)\n",
    "      - max_df=0.95      : ignore terms that appear in >95% of sentences\n",
    "      - min_df=1         : include terms that appear in >=1 sentence\n",
    "    \"\"\"\n",
    "    if not sent_list:\n",
    "        return None\n",
    "\n",
    "    vectorizer = TfidfVectorizer(\n",
    "        stop_words=None,     # Korean: None, English-only: 'english'\n",
    "        ngram_range=(1, 2),  # use unigrams + bigrams\n",
    "        lowercase=True,\n",
    "        max_df=0.95,\n",
    "        min_df=1\n",
    "    )\n",
    "    X = vectorizer.fit_transform(sent_list)\n",
    "    sim = cosine_similarity(X)\n",
    "    return sim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "83ee5528-0e10-41d5-9fa2-74d676c84631",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Step 1 - Similarity matrix (rounded):\n",
      "[[1.   0.05 0.   0.09 0.  ]\n",
      " [0.05 1.   0.09 0.   0.  ]\n",
      " [0.   0.09 1.   0.   0.  ]\n",
      " [0.09 0.   0.   1.   0.  ]\n",
      " [0.   0.   0.   0.   1.  ]]\n"
     ]
    }
   ],
   "source": [
    "# Step 1) Similarity matrix\n",
    "sim_matrix = build_sim_matrix(sents)\n",
    "print(\"\\n✅ Step 1 - Similarity matrix (rounded):\")\n",
    "print(sim_matrix.round(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "35240948-5d66-4509-99e2-fe3319406317",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# 2) TextRank scores (PageRank on similarity graph)\n",
    "#    - Build graph from sim matrix\n",
    "#    - Remove self-loops\n",
    "#    - Optionally prune very small edges (bottom 25%) to reduce noise\n",
    "# ----------------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "127eb188-11dd-4b53-b6e0-79c6209f9dfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rank_sentences(sim):\n",
    "    if sim is None or sim.size == 0:\n",
    "        return []\n",
    "\n",
    "    G = nx.from_numpy_array(sim)\n",
    "    G.remove_edges_from(nx.selfloop_edges(G))\n",
    "\n",
    "    positives = sim[sim > 0]\n",
    "    if positives.size > 0:\n",
    "        thr = np.percentile(positives, 25)  # bottom 25% threshold\n",
    "        for i, j in list(G.edges()):\n",
    "            if sim[i, j] <= thr:\n",
    "                G.remove_edge(i, j)\n",
    "\n",
    "    scores = nx.pagerank(G, alpha=0.85)\n",
    "    return [scores.get(i, 0.0) for i in range(sim.shape[0])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "8425ddf5-9b18-4e0e-86a8-1aa4a46fa85a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Step 2 - TextRank scores:\n",
      "Sentence 1: 0.4082\n",
      "Sentence 2: 0.0612\n",
      "Sentence 3: 0.0612\n",
      "Sentence 4: 0.4082\n",
      "Sentence 5: 0.0612\n"
     ]
    }
   ],
   "source": [
    "# Step 2) TextRank scores\n",
    "scores = rank_sentences(sim_matrix)\n",
    "print(\"\\n✅ Step 2 - TextRank scores:\")\n",
    "for i, score in enumerate(scores, 1):\n",
    "    print(f\"Sentence {i}: {score:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "c9f8c94a-ebd7-4b09-aa65-38b96fee5506",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# 3) Summarize\n",
    "#    - ratio: fraction of sentences to keep (e.g., 0.2 = 20%)\n",
    "#    - min_sent: minimum number of sentences\n",
    "#    - max_sent: optional cap (e.g., fix output to 3 sentences)\n",
    "# -----------------------------\n",
    "def summarize(text, ratio=0.2, min_sent=3, max_sent=None):\n",
    "    sents = sentence_split(text)\n",
    "    if not sents:\n",
    "        return []\n",
    "\n",
    "    sim = build_sim_matrix(sents)\n",
    "    scores = rank_sentences(sim)\n",
    "\n",
    "    n = len(sents)\n",
    "    k = max(min_sent, math.ceil(n * ratio))\n",
    "    if max_sent is not None:\n",
    "        k = min(k, max_sent)\n",
    "\n",
    "    # Take top-k by score, then restore original order for readability\n",
    "    idx_by_score = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]\n",
    "    idx_by_score.sort()\n",
    "    return [sents[i] for i in idx_by_score]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "61261dff-03c2-4769-ba7a-69df72f70030",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Step 3 - Summary result:\n",
      "1. TextRank is a graph-based ranking algorithm for NLP.\n",
      "2. This method is often used for extractive summarization.\n"
     ]
    }
   ],
   "source": [
    "# Step 3) Summarization\n",
    "summary = summarize(demo_text, ratio=0.4, min_sent=2)\n",
    "print(\"\\n✅ Step 3 - Summary result:\")\n",
    "for i, s in enumerate(summary, 1):\n",
    "    print(f\"{i}. {s}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "483cc854-0556-4eb3-b956-ccd9868a0589",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📌 Original text:\n",
      "TextRank is an algorithm for extractive summarization.\n",
      "It uses the PageRank idea to rank sentences by importance.\n",
      "We compute pairwise sentence similarity to build a graph, then select top-scoring sentences.\n",
      "This approach is widely used in simple summarization pipelines.\n",
      "Even beginners can implement a minimal version quickly.\n",
      "\n",
      "📌 Summary result:\n",
      "1. TextRank is an algorithm for extractive summarization.\n",
      "2. It uses the PageRank idea to rank sentences by importance.\n"
     ]
    }
   ],
   "source": [
    "# -----------------------------\n",
    "# Final test run (English text)\n",
    "# -----------------------------\n",
    "text = \"\"\"TextRank is an algorithm for extractive summarization.\n",
    "It uses the PageRank idea to rank sentences by importance.\n",
    "We compute pairwise sentence similarity to build a graph, then select top-scoring sentences.\n",
    "This approach is widely used in simple summarization pipelines.\n",
    "Even beginners can implement a minimal version quickly.\"\"\"\n",
    "\n",
    "summary = summarize(text, ratio=0.4, min_sent=2)\n",
    "\n",
    "print(\"📌 Original text:\")\n",
    "print(text)\n",
    "print(\"\\n📌 Summary result:\")\n",
    "for i, s in enumerate(summary, 1):\n",
    "    print(f\"{i}. {s}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "910d4faa-8ce1-420e-8f8b-fd7495929277",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📌 Input:\n",
      "TextRank is a graph-based ranking algorithm for NLP. It builds a graph of sentences using pairwise similarity. Then it applies PageRank to score sentences by importance. This method is often used for extractive summarization. Even beginners can implement a minimal version quickly.\n",
      "\n",
      "📌 Summary:\n",
      "1. TextRank is a graph-based ranking algorithm for NLP.\n",
      "2. This method is often used for extractive summarization.\n"
     ]
    }
   ],
   "source": [
    "# -----------------------------\n",
    "# 4) Small self-test (you can remove this block)\n",
    "# -----------------------------\n",
    "if __name__ == \"__main__\":\n",
    "    demo = (\n",
    "        \"TextRank is a graph-based ranking algorithm for NLP. \"\n",
    "        \"It builds a graph of sentences using pairwise similarity. \"\n",
    "        \"Then it applies PageRank to score sentences by importance. \"\n",
    "        \"This method is often used for extractive summarization. \"\n",
    "        \"Even beginners can implement a minimal version quickly.\"\n",
    "    )\n",
    "\n",
    "    print(\"📌 Input:\")\n",
    "    print(demo)\n",
    "    print(\"\\n📌 Summary:\")\n",
    "    for i, s in enumerate(summarize(demo, ratio=0.4, min_sent=2), 1):\n",
    "        print(f\"{i}. {s}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12666217-adf1-44aa-9d7c-ae19d430c3ce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (nlp_env)",
   "language": "python",
   "name": "nlp_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
