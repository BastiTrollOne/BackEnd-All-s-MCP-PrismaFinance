from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from app.core.config import settings

# --- Definición del Estado del Grafo ---
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_agent: str
    final_response: str

# --- 1. El Clasificador (El "Portero") ---
async def classifier_node(state: AgentState):
    """Analiza la última pregunta y decide qué agente usar."""
    last_message = state["messages"][-1].content
    
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-needed",
        model=settings.LLM_MODEL_NAME,
        temperature=0
    )
    
    # Prompt de clasificación estricta
    prompt = f"""
    Eres un enrutador inteligente. Tu trabajo es clasificar la intención del usuario en una de estas 3 categorías:
    
    1. 'METABASE': Preguntas sobre DATOS de negocio (ventas, clientes, ingresos, SQL, tablas, dashboards).
    2. 'OPENWEBUI': Preguntas de SISTEMA o ADMINISTRACIÓN (instalar modelos, crear usuarios, ver logs, notas, configuración del servidor).
    3. 'GENERAL': Saludos, preguntas generales o cháchara.
    
    Usuario: "{last_message}"
    
    Responde SOLAMENTE con una palabra: METABASE, OPENWEBUI o GENERAL.
    """
    
    response = await llm.ainvoke(prompt)
    decision = response.content.strip().upper()
    
    # Limpieza básica por si el modelo es muy hablador
    if "METABASE" in decision: return {"next_agent": "metabase"}
    if "OPENWEBUI" in decision: return {"next_agent": "openwebui"}
    return {"next_agent": "general"}

# --- 2. Nodos Ejecutores (Los "Trabajadores") ---

async def metabase_node(state: AgentState, config):
    """Invoca al BrainAgent (Datos)"""
    query = state["messages"][-1].content
    # Recuperamos el agente inyectado en la configuración
    brain_agent = config["configurable"]["brain_agent"]
    
    # Llamada al agente existente
    result = await brain_agent.ainvoke({"input": query})
    return {"final_response": result["output"]}

async def openwebui_node(state: AgentState, config):
    """Invoca al MCPAgent (Admin)"""
    query = state["messages"][-1].content
    mcp_agent = config["configurable"]["mcp_agent"]
    
    # Llamada al agente MCP
    result = await mcp_agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return {"final_response": result["messages"][-1].content}

async def general_node(state: AgentState):
    """Responde preguntas simples directamente"""
    llm = ChatOpenAI(base_url=settings.LM_STUDIO_URL, api_key="not-needed", model=settings.LLM_MODEL_NAME)
    response = await llm.ainvoke(state["messages"])
    return {"final_response": response.content}

# --- 3. Construcción del Grafo ---
def build_orchestrator():
    workflow = StateGraph(AgentState)
    
    # Añadir nodos
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("metabase_agent", metabase_node)
    workflow.add_node("openwebui_agent", openwebui_node)
    workflow.add_node("general_chat", general_node)
    
    # Definir punto de entrada
    workflow.set_entry_point("classifier")
    
    # Definir aristas condicionales (Routing)
    def route(state):
        if state["next_agent"] == "metabase": return "metabase_agent"
        if state["next_agent"] == "openwebui": return "openwebui_agent"
        return "general_chat"

    workflow.add_conditional_edges(
        "classifier",
        route,
        {
            "metabase_agent": "metabase_agent",
            "openwebui_agent": "openwebui_agent",
            "general_chat": "general_chat"
        }
    )
    
    # Todos los agentes terminan el flujo
    workflow.add_edge("metabase_agent", END)
    workflow.add_edge("openwebui_agent", END)
    workflow.add_edge("general_chat", END)
    
    return workflow.compile()