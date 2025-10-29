import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents.base import create_stuff_documents_chain
from langchain.chains.retrieval.base import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ------------------- SETUP -------------------
load_dotenv()

model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ------------------- PDF TEXT EXTRACTION -------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# ------------------- TEXT SPLITTING -------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# ------------------- VECTOR STORE CREATION -------------------
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")


# ------------------- RETRIEVAL CHAIN -------------------
def get_conversation_chain():
    prompt = """Answer the question as detailed as possible from the provided context.
    Include the page number if available.
    If the answer is not present in the PDF, say so clearly.

    Context: {context}
    Question: {input}
    Answer:"""

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["context", "input"]
    )

    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()

    combine_docs_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain
    )
    return retrieval_chain


# ------------------- MAIN CHAT LOGIC -------------------
def get_bot_response(user_question):
    chain = get_conversation_chain()
    response = chain.invoke({"input": user_question})
    return response["answer"]


# ------------------- STREAMLIT UI -------------------
def main():
    st.title("Chat with your PDF")
    st.markdown(
    """
    <style>
    /* ---- Background image ---- */
    .stApp {
        background-image: url("https://unsplash.com/illustrations/a-black-and-white-drawing-of-a-woman-wearing-sunglasses-U9gO48rX4NQ");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* ---- Soft white overlay for readability ---- */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.75); /* Adjust for brightness */
        z-index: 0;
    }

    /* ---- Bring app content above overlay ---- */
    .stApp > div:first-child {
        position: relative;
        z-index: 1;
    }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.9); /* White semi-transparent bubble */
        color: black !important;               /* Black text inside bubbles */
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
    }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        color: black !important;
    }

    /* ---- General text color ---- */
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )



    # Sidebar for PDF Upload
    with st.sidebar:
        st.subheader("📁 Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                pdf_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(pdf_text)
                get_vectorstore(text_chunks)
                st.success("Documents processed successfully! You can start chatting below.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    user_question = st.chat_input("Ask something about your PDF...")

    if user_question:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_bot_response(user_question)
                st.markdown(response)

        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": response})


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    main()
