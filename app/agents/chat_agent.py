from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

# 1. Define el LLM que se conectará a LM Studio
llm = ChatOpenAI(
    api_key="not-needed",
    base_url=settings.LM_STUDIO_URL,
    model=settings.LLM_MODEL_NAME,
    temperature=0, # A standard value for creative and conversational responses
    timeout=120 # 2-minute timeout for conversational responses
)

# 2. Define un prompt simple para conversación
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente conversacional amigable y servicial."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
])

# 3. Crea la cadena (chain) que une el prompt, el LLM y el parser de salida
chat_agent = prompt | llm | StrOutputParser()