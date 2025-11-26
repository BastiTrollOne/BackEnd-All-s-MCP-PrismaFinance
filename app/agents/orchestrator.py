import logging
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from app.core.config import settings

# Configurar Logger
logger = logging.getLogger("Orchestrator")

# --- Definición del Estado del Grafo ---
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_agent: str
    final_response: str

# --- 1. El Clasificador (El "Portero") ---
async def classifier_node(state: AgentState):
    """
    Analiza la última pregunta y decide qué agente usar.
    Prioriza palabras clave explícitas antes de usar el LLM.
    """
    last_message_obj = state["messages"][-1]
    last_message = last_message_obj.content
    msg_lower = last_message.lower()
    
    logger.info(f"⚡ [ORCHESTRATOR] Analizando: '{last_message}'")

    # --- 🚀 ATAJOS DETERMINISTAS (Reglas fijas) ---
    # Si el usuario menciona explícitamente la herramienta, obedecemos sin pensar.
    
    if "metabase" in msg_lower:
        logger.info("🚀 [ATAJO] Palabra clave 'metabase' detectada -> Metabase Agent")
        return {"next_agent": "metabase"}
        
    if "open webui" in msg_lower or "openwebui" in msg_lower:
        logger.info("🚀 [ATAJO] Palabra clave 'open webui' detectada -> OpenWebUI Agent")
        return {"next_agent": "openwebui"}
    
    # ----------------------------------------------

    # Si no hay palabras clave, usamos la IA para entender la intención
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-needed",
        model=settings.LLM_MODEL_NAME,
        temperature=0
    )
    
    prompt = f"""
    Eres un enrutador inteligente. Tu trabajo es clasificar la intención del usuario en una de estas 3 categorías:
    
    1. 'METABASE': Preguntas sobre DATOS de negocio (ventas, clientes, ingresos, SQL, tablas, dashboards, gráficas).
    2. 'OPENWEBUI': Preguntas técnicas sobre el SISTEMA (usuarios, modelos, configuración, logs, API, servidor).
    3. 'GENERAL': Saludos, preguntas generales o cháchara.
    
    Usuario: "{last_message}"
    
    Responde SOLAMENTE con una palabra: METABASE, OPENWEBUI o GENERAL.
    """
    
    response = await llm.ainvoke(prompt)
    decision = response.content.strip().upper()
    
    logger.info(f"🚦 [IA DECISION] El modelo eligió: {decision}")

    # Mapeo de la decisión de la IA
    if "METABASE" in decision: return {"next_agent": "metabase"}
    if "OPENWEBUI" in decision: return {"next_agent": "openwebui"}
    return {"next_agent": "general"}

# --- 2. Nodos Ejecutores (Los "Trabajadores") ---

async def metabase_node(state: AgentState, config):
    logger.info("📊 [EJECUTANDO] Metabase Agent...")
    query = state["messages"][-1].content
    brain_agent = config["configurable"]["brain_agent"]
    
    if not brain_agent:
        return {"final_response": "❌ Error: El agente de Metabase no está inicializado."}

    result = await brain_agent.ainvoke({"input": query})
    return {"final_response": result["output"]}

async def openwebui_node(state: AgentState, config):
    logger.info("🔧 [EJECUTANDO] Open WebUI Agent...")
    query = state["messages"][-1].content
    mcp_agent = config["configurable"]["mcp_agent"]

    if mcp_agent is None:
        return {
            "final_response": (
                "⚠️ **Servicio no disponible**: No pude conectar con el agente de administración (Open WebUI) "
                "durante el inicio del sistema. Por favor verifica que el servidor MCP esté corriendo en el puerto 9001."
            )
        }
    
    # Llamada al agente MCP
    result = await mcp_agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return {"final_response": result["output"]}

async def general_node(state: AgentState):
    logger.info("💬 [EJECUTANDO] Chat General...")
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