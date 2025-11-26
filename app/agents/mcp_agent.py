import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from app.core.config import settings 

logger = logging.getLogger("MCPWorker")

# --- 1. PROMPT DEL SISTEMA ---
# Definimos el prompt como una plantilla de chat estándar de LangChain
system_message = """
Eres un **Ingeniero de Operaciones de Open WebUI (DevOps)**.
Estás conectado a un servidor MCP que actúa como una "puerta de enlace" a la API interna.

⛔ **ADVERTENCIA CRÍTICA DE ARQUITECTURA**:
NO tienes herramientas específicas pre-cargadas (como `create_user` o `delete_model`).
SOLO tienes 3 herramientas genéricas de **descubrimiento y ejecución**.

🛠️ **TU PROTOCOLO DE 3 PASOS (OBLIGATORIO):**

1.  **🔍 EXPLORACIÓN (`list_available_openwebui_operations`)**:
    -   JAMÁS intentes ejecutar un comando sin buscarlo antes.
    -   Usa esta herramienta filtrando por `category` (ej: 'User', 'Model', 'Auth', 'Chat') para encontrar la `operation_id` correcta.
    -   Si no encuentras nada, prueba sin filtros.

2.  **📖 INSPECCIÓN (`get_operation_details`)**:
    -   Una vez tengas la `operation_id` (ej: `users_get_all`), **DEBES** usar esta herramienta.
    -   Tu objetivo es leer el esquema JSON (argumentos requeridos, estructura del payload) antes de intentar llamar a la API.
    -   *No adivines los parámetros.* Míralos en la descripción que te devuelve esta herramienta.

3.  **🚀 EJECUCIÓN (`call_openwebui_api`)**:
    -   Solo ahora puedes ejecutar la acción.
    -   Usa la `operation_id` validada y pasa los argumentos exactamente como los viste en el paso de inspección.

**MANEJO DE ERRORES:**
- Si `call_openwebui_api` devuelve un error 400/422, SIGNIFICA que los argumentos están mal. Vuelve al paso 2 (Inspección).
"""

# --- 2. ADAPTADOR DE HERRAMIENTAS ---
def mcp_to_langchain_tool(mcp_tool, session):
    """Convierte una herramienta cruda de MCP a algo que LangChain puede ejecutar."""
    async def wrapped_tool(**kwargs):
        logger.info(f"🛠️  MCP Worker ejecutando: {mcp_tool.name} {kwargs}")
        try:
            result = await session.call_tool(mcp_tool.name, arguments=kwargs)
            
            # Extraer texto limpio del resultado MCP
            if result.content and hasattr(result.content[0], 'text'):
                return result.content[0].text
            return str(result)
        except Exception as e:
            logger.error(f"Error ejecutando herramienta MCP: {e}")
            return f"Error: {str(e)}"

    return StructuredTool.from_function(
        func=None,
        coroutine=wrapped_tool,
        name=mcp_tool.name,
        description=mcp_tool.description or "Herramienta de Open WebUI",
    )

# --- 3. CONSTRUCTOR DEL AGENTE ---
async def build_mcp_worker_agent(session):
    """
    Construye el agente usando AgentExecutor (Método Clásico).
    Evita el uso de langgraph.prebuilt.create_react_agent para prevenir errores de argumentos.
    """
    logger.info("Construyendo MCP Worker Agent (Modo Clásico)...")

    # A. Herramientas
    logger.info("Solicitando lista de herramientas al servidor MCP...")
    mcp_tools_list = await session.list_tools()
    worker_tools = [mcp_to_langchain_tool(t, session) for t in mcp_tools_list.tools]
    logger.info(f"Herramientas cargadas: {[t.name for t in worker_tools]}")

    # B. LLM
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-needed",
        model=settings.LLM_MODEL_NAME,
        temperature=0,
    )

    # C. Prompt Template (Requerido para create_openai_tools_agent)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{messages}"), # El orquestador envía una lista de mensajes aquí
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Memoria intermedia obligatoria
    ])

    # D. Crear el Agente y el Executor
    agent = create_openai_tools_agent(llm, worker_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=worker_tools,
        verbose=True,
        max_iterations=15, # Darle espacio para pensar y reintentar pasos
        handle_parsing_errors=True
    )

    return agent_executor