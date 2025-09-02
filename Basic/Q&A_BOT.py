import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser   
load_dotenv()

st.set_page_config(
    page_title="Q&A Bot", 
    page_icon="🤖", 
    layout="centered"
)

st.title("Q&A Bot")
st.write("Ask me anything!")

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
parser = StrOutputParser()
chain = prompt | llm | parser

user_input = st.text_input("Enter your question:", placeholder="Type your question here...")

# Process and display response
if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
    
    st.write("**Your Question:**", user_input)
    st.write("**Response:**")
    st.write(response)

