import hashlib
from pathlib import Path

from pypdf import PdfReader

from retrievalbench.model import Document


# generate unique document_id based on the content of the document
def make_document_id(file_path: Path) -> str:
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def extract_file(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8")


# load the corpus and convert into the list of Document object
def load_corpus(corpus_dir: str) -> list[Document]:
    documents = []
    for file_path in Path(corpus_dir).glob("*"):
        if file_path.suffix not in [".pdf", ".md", ".txt"]:
            continue

        doc_id = make_document_id(file_path)
        documents.append(
            Document(
                id=doc_id,
                source_path=str(file_path),
                title=file_path.stem,
                text=extract_file(file_path),
            )
        )
    return documents
