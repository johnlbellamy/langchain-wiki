import os

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI

from query import get_wiki_entry
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import Pinecone as PineconeStore
#from dotenv import load_dotenv

#load_dotenv()

open_key = os.environ.get("OPEN_API_KEY")
pine_key = os.environ.get("PINE_API_KEY")
os.environ["PINECONE_API_KEY"] = pine_key
embeddings = OpenAIEmbeddings(openai_api_key=open_key, model="text-embedding-3-large", dimensions=1024)


def get_tokens(wiki_articles):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=0
    )
    tokens = splitter.split_text(wiki_articles)
    return tokens


def get_qa(tokens):
    search = PineconeStore.from_texts(tokens,
                                      embedding=embeddings,
                                      index_name="wikipedia")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=search.as_retriever(),  # this calls the vector store for relevant context
        return_source_documents=True
    )
    return qa


if __name__ == "__main__":
    query = "What cases did the supreme court hear in 2024"

    document = get_wiki_entry(query)
    print(document)
    tokens = get_tokens(document)

    qa = get_qa(tokens)
    # prompt_template = PromptTemplate()

    result = qa({"query": query})
    print(result['result'])
