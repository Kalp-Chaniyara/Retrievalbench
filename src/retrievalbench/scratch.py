# scratch.py  (throwaway, don't commit)
from retrievalbench.ingest.chunkers import FixedSizeChunker
from retrievalbench.ingest.loader import load_corpus

docs = load_corpus("/Users/apple/CODING/Retrievalbench/corpora/sample_data1")
print(f"loaded {len(docs)} docs")

chunker = FixedSizeChunker(size=200, overlap=40)
for d in docs:
    chunks = chunker.chunk(d)
    print(f"this is the chunk of the {d.title}: {chunks}")
