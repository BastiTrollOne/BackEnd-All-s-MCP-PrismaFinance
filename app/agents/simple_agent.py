from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

from app.core.config import settings

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Using a safer eval method
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

llm = ChatOpenAI(
    api_key="not-needed",
    base_url=settings.LM_STUDIO_URL,
    model=settings.LLM_MODEL_NAME
)
tools = [calculator]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Tienes acceso a una herramienta de calculadora."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent instance
simple_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
