import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="Conversational Bot", page_icon="🦈", layout="wide")

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 Conversational Math Assistant")
st.markdown("---")

# Initialize the chat model
@st.cache_resource
def load_model():
    return ChatOpenAI()

chat = load_model()

# Initialize session state for messages
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a helpful Math assistant. Be friendly and conversational.")
    ]

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def get_ChatModel_response(question):
    # Add user message
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    
    # Get AI response
    answer = chat(st.session_state['flowmessages'])
    
    # Add AI message to flow
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    
    # Add to chat history for display
    st.session_state['chat_history'].append({"role": "user", "content": question})
    st.session_state['chat_history'].append({"role": "assistant", "content": answer.content})
    
    return answer.content

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state['chat_history']:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# Chat input at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me anything about math...", 
            key="user_input",
            placeholder="Type your message here..."
        )
    
    with col2:
        submit = st.form_submit_button("Send 📤")

# Process the input
if submit and user_input:
    with st.spinner("Thinking..."):
        response = get_ChatModel_response(user_input)
    
    # Rerun to update the chat display
    st.rerun()

# Add a clear chat button
if st.session_state['chat_history']:
    if st.button("🗑️ Clear Chat"):
        st.session_state['chat_history'] = []
        st.session_state['flowmessages'] = [
            SystemMessage(content="You are a helpful Math assistant. Be friendly and conversational.")
        ]
        st.rerun()

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    - Ask any math question
    - Get step-by-step solutions
    - Have a natural conversation
    - Use the clear button to start fresh
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Be specific with your questions
    - Ask for explanations if needed
    - Try different problem types
    """)