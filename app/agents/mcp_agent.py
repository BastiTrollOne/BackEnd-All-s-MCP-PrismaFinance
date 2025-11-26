import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from app.core.config import settings 

logger = logging.getLogger("MCPWorker")

# --- 1. PROMPT DEL SISTEMA ---
system_message = """
Eres un **Ingeniero de Operaciones de Open WebUI (DevOps)**.
Estás conectado a un servidor MCP que actúa como una "puerta de enlace" a la API interna.

⛔ **ADVERTENCIA CRÍTICA**: NO tienes herramientas pre-cargadas. Dependes de las herramientas de descubrimiento.

🛠️ **TU PROTOCOLO DE TRABAJO INTELIGENTE:**

**CASO A: NO CONOCES EL ID DE LA OPERACIÓN**
1.  **🔍 EXPLORACIÓN (`list_available_openwebui_operations`)**:
    -   Usa esta herramienta filtrando por `category` (ej: 'User', 'Model') para encontrar el ID.
2.  **📖 INSPECCIÓN (`get_operation_details`)**:
    -   Usa el ID encontrado para aprender sus argumentos (JSON Schema).
3.  **🚀 EJECUCIÓN (`call_openwebui_api`)**:
    -   Ejecuta la acción.

**CASO B: YA TIENES EL ID DE LA OPERACIÓN** (Ej: El usuario te lo dio explícitamente)
1.  **⏭️ OMITE LA EXPLORACIÓN**: No pierdas tiempo listando categorías.
2.  **📖 INSPECCIÓN DIRECTA (`get_operation_details`)**:
    -   Llama INMEDIATAMENTE a esta herramienta con el `operation_id` que te dieron.
    -   Verifica qué argumentos requiere.
3.  **🚀 EJECUCIÓN (`call_openwebui_api`)**:
    -   Ejecuta la operación con los argumentos confirmados.

**MANEJO DE ERRORES:**
- Si `get_operation_details` falla diciendo "ID no encontrado", entonces (y solo entonces) vuelve al CASO A paso 1 para buscar el ID correcto.
- Si `call_openwebui_api` devuelve error 4xx, revisa tus argumentos.

5. **CRITERIO DE FINALIZACIÓN (MUY IMPORTANTE):**
- Tu misión termina INMEDIATAMENTE después de usar `call_openwebui_api` y recibir una respuesta (sea éxito o error).
- **NO** intentes verificar el resultado llamando a otra API.
- **NO** vuelvas a listar operaciones.
- Genera tu respuesta final basada en el JSON que recibiste y DETENTE.

4. **FORMATO DE RESPUESTA (OBLIGATORIO)**
- Al final de tu respuesta, firma con las herramientas usadas:
---
🛠 **Herramientas/Secuencia usada:** [Herramienta_1] > [Herramienta_2] (o "Ninguna")
"""

# --- 2. ADAPTADOR DE HERRAMIENTAS ---
def mcp_to_langchain_tool(mcp_tool, session):
    """Convierte una herramienta cruda de MCP a algo que LangChain puede ejecutar."""
    async def wrapped_tool(**kwargs):
        logger.info(f"🛠️  MCP Worker ejecutando: {mcp_tool.name} {kwargs}")
        try:
            result = await session.call_tool(mcp_tool.name, arguments=kwargs)
            
            # Extraer texto limpio del resultado MCP si es necesario
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

    # C. Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{messages}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # D. Crear el Agente y el Executor
    agent = create_openai_tools_agent(llm, worker_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=worker_tools,
        verbose=True,
        max_iterations=8, # Reducido para forzar paradas tempranas si se confunde
        handle_parsing_errors=True
    )

    return agent_executor