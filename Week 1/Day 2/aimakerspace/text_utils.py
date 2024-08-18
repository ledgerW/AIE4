import os
from pypdf import PdfReader
from typing import List


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_txt_file()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_txt_file(self, path: os.path = None):
        _path = path if path else self.path
        
        with open(_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    
    def load_pdf_file(self, path: os.path = None):
        _path = path if path else self.path
        
        reader = PdfReader(_path)
        num_pages = len(reader.pages)

        doc_text = ""
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            doc_text = doc_text + page_text

        self.documents.append(doc_text)
            

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    self.load_txt_file(path=os.path.join(root, file))
                elif file.endswith(".pdf"):
                    self.load_pdf_file(path=os.path.join(root, file))
                    #with open(
                    #    os.path.join(root, file), "r", encoding=self.encoding
                    #) as f:
                    #    self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, doc_meta: dict) -> List[dict]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = {'text': text[i : i + self.chunk_size], **doc_meta}
            chunks.append(chunk)
        return chunks

    def split_texts(self, texts: List[str], docs_meta: List[dict] = None) -> List[dict]:
        chunks = []
        for text, doc_meta in zip(texts, docs_meta):
            chunks.extend(self.split(text, doc_meta))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
