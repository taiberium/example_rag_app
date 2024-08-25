from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import asyncio
import re

qa_chain = None  # Added this line to define qa_chain as a global variable

def create_rag_app(md_file_path: str, model_path: str) -> RetrievalQA:
    # Load the markdown file
    loader: TextLoader = TextLoader(md_file_path)
    documents: List[Document] = loader.load()

    # Split the text into chunks
    text_splitter: CharacterTextSplitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts: List[Document] = text_splitter.split_documents(documents)

    # Create embeddings and store them in a Chroma vector store
    embeddings: GPT4AllEmbeddings = GPT4AllEmbeddings()
    db: Chroma = Chroma.from_documents(texts, embeddings)

    # Initialize the GPT4All model with LLama
    llm: GPT4All = GPT4All(model=model_path, backend="llama", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

    # Create a retrieval-based QA chain
    prompt_template = """Use the following information to answer the user's question. Respond briefly, with a single sentence, without using hashtags or additional symbols:

{context}

Question: {question}
Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

class CustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # print(token, end="", flush=True)

async def generate_answer(question):
    global qa_chain
    handler = CustomHandler()
    qa_chain.combine_documents_chain.llm_chain.llm.callbacks = [handler]
    
    await qa_chain.ainvoke(question)
    
    # Извлекаем первое предложение из ответа
    full_answer = handler.text.strip()
    first_sentence = re.split(r'(?<=[.!?])\s', full_answer, maxsplit=1)[0]
    
    return first_sentence.strip()

async def main():
    global qa_chain
    qa_chain = create_rag_app("example_rag_app/test_rag_file.md", "/Users/taiberium/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf")
    
    question = input("Введите ваш вопрос: ")
    print("Генерация ответа...")
    answer = await generate_answer(question)
    print("\nОтвет:", answer)

if __name__ == "__main__":
    asyncio.run(main())