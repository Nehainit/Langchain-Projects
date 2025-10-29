from langchain_community.document_loaders import NotionDBLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# loader = NotionDirectoryLoader("Notion_DB")
loader = NotionDBLoader(
    integration_token=os.getenv("NOTION_TOKEN"),
    database_id=os.getenv("database_id"),
    request_timeout_sec=30,  # optional, defaults to 10
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
new_db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
user_question = input("Enter your question: ")
docs = new_db.similarity_search(user_question)
prompt = PromptTemplate(
    template="""
    You are a personal Note Taker bot.Please answer following 
    Question:{user_question} 
    based on these context: {docs}.
    If the topic has not been covered in the context,then dont add the extra topic.
    """)

model=ChatOpenAI( temperature=0)
chain = prompt | model
response = chain.invoke({"user_question": user_question, "docs": docs})
print(response.content)

