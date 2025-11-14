from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# embed query
# vector = embeddings.embed_query("hello, world!")
# print(vector[:5])

# embed docs

documents = ["India is in Asia", "Virat is the best india cricket player", "Barcelona is a city in spain"]

result = embeddings.embed_documents(documents)

# Retrival
# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

# text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    documents,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("who is the best cricket player")

# show the retrieved document's content
print(retrieved_documents[0])