# gmail_send_mail.py

from dotenv import load_dotenv
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ----------------------------------------------------
# 1️⃣ Load environment variables (OPENAI_API_KEY etc.)
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# 2️⃣ Gmail Authentication
# ----------------------------------------------------
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
tools = toolkit.get_tools()

print("✅ Gmail tools loaded:")
for t in tools:
    print("-", t.name)

# ----------------------------------------------------
# 3️⃣ Initialize LLM
# ----------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----------------------------------------------------
# 4️⃣ Define system + user prompt
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Gmail assistant that can read, draft, and send emails."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ----------------------------------------------------
# 5️⃣ Create agent & executor
# ----------------------------------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------------------------------
# 6️⃣ Example — Send an email
# ----------------------------------------------------
example_query = """
Send an email to "nehadubey1021@gmail.com"
Subject: Thank you for the meeting
Body: Hi, it was great catching up over coffee today! Let's stay in touch.
"""

response = agent_executor.invoke({"input": example_query})
print("\n📧 Email Sent Response:")
print(response) 
