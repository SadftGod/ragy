import os
from dotenv import load_dotenv

# Загружаем переменные из .env (ключ kry или OPENAI_API_KEY)
load_dotenv()

from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def build_faiss_index(data_path: str, index_path: str):
    # Загружаем все .txt из папки
    loader = DirectoryLoader(data_path, glob="**/*.txt")
    raw_docs = loader.load()
    if not raw_docs:
        raise RuntimeError(f"Не найдено .txt в папке {data_path}. Проверьте, что файлы лежат там.")

    # Семантическое разбиение на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_docs)

    # Явно передаём ключ API
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("kry") or os.getenv("OPENAI_API_KEY"))
    faiss_index = FAISS.from_documents(docs, embeddings)

    # Сохраняем индекс на диск
    faiss_index.save_local(index_path)
    print(f"FAISS index saved to {index_path}")

    return faiss_index


def load_faiss_index(index_path: str):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("kry") or os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(index_path, embeddings)


def create_qa_chain(faiss_index):
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("kry") or os.getenv("OPENAI_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    DATA_PATH = "./data"
    INDEX_PATH = "./faiss_index"

    # Построить индекс, если его нет, иначе загрузить
    if not os.path.exists(INDEX_PATH):
        index = build_faiss_index(DATA_PATH, INDEX_PATH)
    else:
        index = load_faiss_index(INDEX_PATH)

    qa = create_qa_chain(index)

    print("RAG system ready. Ask your questions! (empty input to exit)")
    while True:
        query = input("You: ").strip()
        if not query:
            break
        result = qa({"question": query})
        answer = result.get("answer")
        sources = result.get("source_documents", [])

        print(f"\nAssistant: {answer}\n")
        if sources:
            print("Источники:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'unknown')}")
            print()
