import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from app.core.config import settings 

logger = logging.getLogger("MCPWorker")

# --- 1. PROMPT DEL SISTEMA (OPTIMIZADO) ---
system_message = """
Eres un **Ingeniero de Operaciones de Open WebUI (DevOps)** experto en APIs.
Estás conectado a un servidor MCP que actúa como "puerta de enlace". No tienes herramientas pre-cargadas; debes descubrirlas.

🛠️ **TU PROTOCOLO DE INVESTIGACIÓN (FLUJO EN CASCADA):**

**FASE 1: 🧭 ORIENTACIÓN (¿Dónde busco?)**
- Si NO sabes la categoría exacta, llama a `list_available_openwebui_operations()` **SIN ARGUMENTOS**.
- **CRÍTICO:** La respuesta te dará una lista de categorías válidas (ej: `['User/Auth/Groups', 'System/Config', ...]`).
- **ACCIÓN:** Selecciona la categoría correcta y **CÓPIALA EXACTAMENTE** (incluyendo mayúsculas, barras `/` y espacios).
    - ❌ INCORRECTO: category='User'
    - ✅ CORRECTO: category='User/Auth/Groups'

**FASE 2: 🔍 EXPLORACIÓN (¿Qué herramientas hay?)**
- Con el nombre EXACTO de la categoría:
- **ACCIÓN:** Llama a `list_available_openwebui_operations(category='NOMBRE_EXACTO')`.
- Si recibes una advertencia ("No se encontraron operaciones..."), **LEE** la lista de válidas que te devuelve el error y reintenta inmediatamente con el nombre correcto.

**FASE 3: 🕵️ SELECCIÓN Y VALIDACIÓN**
- Selecciona la `operation_id` más prometedora de la lista.
- **ACCIÓN:** Llama a `get_operation_details(operation_id='...')`.
- **DECISIÓN:**
    - ✅ **SI SIRVE:** Pasa a la FASE 4.
    - ❌ **NO SIRVE:** Vuelve a la lista de la FASE 2 y prueba la siguiente operación.

**FASE 4: 🚀 EJECUCIÓN**
- **ACCIÓN:** Llama a `call_openwebui_api(operation_id='...', arguments={{...}})`.

---
**CASO ESPECIAL: ATAJO (ID CONOCIDO)**
- Si ya tienes el `operation_id` exacto, salta directo a la **FASE 3** (Inspección).

---
**REGLAS DE ORO:**
1. **CRITERIO DE FINALIZACIÓN:** Tu misión termina INMEDIATAMENTE después de recibir la respuesta exitosa de `call_openwebui_api`. Genera tu respuesta final basada en ese JSON y **DETENTE**.
2. **NO VERIFIQUES:** No llames a ninguna API extra para confirmar. Confía en el código 200 OK.

**FORMATO DE RESPUESTA (OBLIGATORIO)**
Al final, firma con la secuencia real:
---
🛠 **Herramientas/Secuencia usada:** [Herramienta_1] > [Herramienta_2] > ...
"""

# --- 2. ADAPTADOR DE HERRAMIENTAS ---
def mcp_to_langchain_tool(mcp_tool, session):
    """Convierte una herramienta cruda de MCP a algo que LangChain puede ejecutar."""
    async def wrapped_tool(**kwargs):
        logger.info(f"🛠️  MCP Worker ejecutando: {mcp_tool.name} {kwargs}")
        try:
            result = await session.call_tool(mcp_tool.name, arguments=kwargs)
            
            # Limpieza de respuesta para ahorrar tokens y evitar bucles por texto excesivo
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
        max_iterations=5, # Aumentado a 20 para dar margen a la exploración sin ser infinito
        handle_parsing_errors=True
    )

    return agent_executor