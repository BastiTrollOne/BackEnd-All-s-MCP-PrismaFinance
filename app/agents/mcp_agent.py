import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from app.core.config import settings 

logger = logging.getLogger("MCPWorker")

# --- 1. PROMPT DEL SISTEMA (Personalidad del Especialista) ---
# Este prompt define estrictamente que este agente es un técnico de Open WebUI.
MCP_WORKER_PROMPT = """
Eres un Administrador Avanzado de Open WebUI.
Tu trabajo es gestionar el sistema usando EXCLUSIVAMENTE las herramientas disponibles.

TU MISIÓN:
Recibirás instrucciones técnicas del Agente Principal (Brain). Debes ejecutarlas usando las herramientas de la API.

REGLAS DE ORO:
1. NO inventes respuestas. Si no sabes algo, usa las herramientas.
2. Tienes 3 herramientas maestras. Úsalas en este orden lógico:
   - PASO A: Usa 'list_available_openwebui_operations' para descubrir comandos (ej: category='User').
   - PASO B: Usa 'get_operation_details' para aprender los argumentos necesarios.
   - PASO C: Usa 'call_openwebui_api' para ejecutar la acción.

EJEMPLO: Si te piden "Lista los modelos":
1. Llama a list...(category="Model (General)")
2. Encuentra la operación correcta.
3. Ejecuta call...(operation_id=...)
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
        coroutine=wrapped_tool, # Importante: MCP es asíncrono
        name=mcp_tool.name,
        description=mcp_tool.description or "Herramienta de Open WebUI",
    )

# --- 3. CONSTRUCTOR DEL AGENTE (Factory Function) ---
async def build_mcp_worker_agent(session):
    """
    Esta función es llamada por main.py.
    Recibe la sesión activa de MCP y devuelve el agente listo para trabajar.
    """
    logger.info("Construyendo MCP Worker Agent...")

    # A. Obtener herramientas del servidor p.py a través de la sesión
    logger.info("Solicitando lista de herramientas al servidor MCP...")
    mcp_tools_list = await session.list_tools()
    
    # B. Convertirlas a formato LangChain
    worker_tools = [mcp_to_langchain_tool(t, session) for t in mcp_tools_list.tools]
    logger.info(f"Herramientas cargadas en Worker: {[t.name for t in worker_tools]}")

    # C. Configurar el LLM
    # Usamos la configuración centralizada de tu app
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-needed",
        model=settings.LLM_MODEL_NAME,
        temperature=0, # Precisión máxima para uso de herramientas
    )

    # D. Crear y devolver el Agente ReAct
    # 'state_modifier' inyecta el Prompt del Sistema al inicio
    return create_react_agent(llm, worker_tools, state_modifier=MCP_WORKER_PROMPT)