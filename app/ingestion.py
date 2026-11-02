from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample_docs" / "api_guide.txt"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    """Read a UTF-8 text/Markdown file or extract text from a PDF."""
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. Supported: .txt, .md, .pdf"
        )

    if extension == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(
            f"No extractable text found in document: {path}. "
            "Scanned PDFs may require OCR."
        )

    return text


def split_document(
    text: str,
    source: str = "api_guide.txt",
) -> list[Document]:
    """Split text into overlapping chunks with source metadata."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot split empty document text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    document = Document(
        page_content=text,
        metadata={"source": source},
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    return chunks


if __name__ == "__main__":
    text = load_document(DOCUMENT_PATH)
    chunks = split_document(text, source=str(DOCUMENT_PATH.resolve()))

    print(f"Loaded document: {DOCUMENT_PATH}")
    print(f"Generated {len(chunks)} chunks.")

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index} ---")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content)
