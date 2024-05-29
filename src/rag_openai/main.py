import dotenv
from pathlib import Path
from time import perf_counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv(override=False)


class RAGOpenAI:
    MODEL_NAME = "gpt-4o"

    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent
        self._input_files_dir = self._base_dir / "files" / "input"
        self._output_files_dir = self._base_dir / "files" / "output"
        self._document_path = self._input_files_dir / "tesla-model-s.pdf"
        self._vector_store_dir = self._output_files_dir / "vector_store"

    def start(self):
        start_time = perf_counter()

        vector_store = self._get_vector_store()
        retriever = vector_store.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model=self.MODEL_NAME)

        chain_context = {
            "context": retriever | self._format_docs,
            "question": RunnablePassthrough(),
        }
        chain = chain_context | prompt | llm | StrOutputParser()

        print("=" * 50)
        question = "Tell me about the noise inside the cabin"
        answer = chain.invoke("Tell me about the noise inside the cabin")
        print(f"Question:\n{question}")
        print("")
        print(f"Answer:\n{answer}")

        end_time = perf_counter()
        execution_time = end_time - start_time
        print("-" * 50)
        print(f"Execution time: {execution_time:.3f}s")

    def _load_document(self):
        loader = PyPDFLoader(self._document_path)

        return loader.load()

    def _get_chunks(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        return splitter.split_documents(docs)

    def _get_vector_store(self):
        must_create = False

        if not self._vector_store_dir.exists():
            self._vector_store_dir.mkdir(parents=True)
            must_create = True
        elif not any(self._vector_store_dir.iterdir()):
            must_create = True

        embeddings_model = OpenAIEmbeddings()

        if must_create:
            docs = self._load_document()
            chunks = self._get_chunks(docs)

            if chunks is None:
                raise ValueError("Can't save vector store, invalid chunks.")

            vector_db = FAISS.from_documents(chunks, embeddings_model)
            vector_db.save_local(self._vector_store_dir.as_posix())
        else:
            vector_db = FAISS.load_local(
                self._vector_store_dir.as_posix(),
                embeddings_model,
                allow_dangerous_deserialization=True,
            )

        return vector_db

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("")
    rag = RAGOpenAI()
    rag.start()
    print("")


if __name__ == "__main__":
    main()
