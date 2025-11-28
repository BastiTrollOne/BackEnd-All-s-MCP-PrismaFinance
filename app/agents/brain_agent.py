import logging
import re
import json
from typing import List, Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field

# Ajusta estos imports a tu estructura real
from app.core.config import settings 
# Importamos utilidades comunes para evitar dependencias circulares si fuera necesario
from app.utils import normalize_sql, enforce_limit

brain_logger = logging.getLogger("LangChainBrainAgent")

# --- 1. ESQUEMAS DE ENTRADA (Input Schemas) ---

class MetabaseToolInput(BaseModel):
    operation_id: str = Field(..., description="El ID técnico de la operación (ej: 'get_api_database')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Parámetros query/body/path.")

class RunSqlInput(BaseModel):
    database_id: int = Field(..., description="ID numérico de la base de datos (obtenlo primero listando las DBs).")
    sql_query: str = Field(..., description="Consulta SQL válida y ejecutable.")

class CreateCardInput(BaseModel):
    database_id: int = Field(..., description="ID numérico de la base de datos.")
    sql_query: str = Field(..., description="Consulta SQL para la tarjeta.")
    name: str = Field(..., description="Nombre descriptivo para la tarjeta.")
    description: str = Field(default="", description="Descripción opcional.")
    display: str = Field(default="table", description="Tipo de visualización: 'table', 'line', 'bar', 'row', 'scalar'.")

class XRayInput(BaseModel):
    entity_id: int = Field(..., description="El ID numérico de la tabla/modelo/pregunta.")
    entity_type: str = Field(default="table", description="Tipo de entidad: 'table', 'question', 'model'.")

# --- NUEVO ESQUEMA AÑADIDO ---
class ExecuteCardInput(BaseModel):
    card_id: int = Field(..., description="El ID numérico de la Card/Pregunta que se quiere ejecutar.")

# --- 2. ADAPTADOR DE HERRAMIENTAS (MCP -> LangChain) ---
def mcp_to_langchain_tool(mcp_tool, session):
    
    # Wrapper genérico para ejecutar la herramienta
    async def wrapped_tool(**kwargs):
        brain_logger.info(f"🛠️ Tool Call: {mcp_tool.name} | Args: {kwargs}")
        try:
            # En el caso de call_metabase_api, reestructuramos los args
            if mcp_tool.name == "call_metabase_api" and "operation_id" in kwargs:
                args = {"operation_id": kwargs["operation_id"], "payload": kwargs.get("payload", {})}
            else:
                args = kwargs

            result = await session.call_tool(mcp_tool.name, arguments=args)
            
            # Extraer texto del resultado MCP
            if hasattr(result, 'content') and result.content:
                return next((c.text for c in result.content if hasattr(c, 'text')), str(result))
            return str(result)
        except Exception as e:
            return f"Error Tool: {str(e)}"

    # --- ASIGNACIÓN DE ESQUEMAS ---
    
    # 1. Herramienta para Ejecutar Cards Existentes (NUEVO)
    if mcp_tool.name == "execute_saved_card":
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description="Ejecuta una pregunta/card existente y devuelve los datos actualizados.",
            args_schema=ExecuteCardInput
        )

    # 2. Herramienta para SQL Rápido
    elif mcp_tool.name == "run_sql_query":
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description="Ejecuta SQL ad-hoc para responder preguntas puntuales. Devuelve datos JSON.",
            args_schema=RunSqlInput
        )

    # 3. Herramienta para Crear Cards
    elif mcp_tool.name == "create_sql_card":
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description="Crea y GUARDA una nueva pregunta/card en Metabase.",
            args_schema=CreateCardInput
        )

    # 4. Herramienta de Dashboards Automáticos (X-Rays)
    elif mcp_tool.name == "generate_and_save_xray":
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description="Genera un DASHBOARD completo automáticamente basado en una tabla.",
            args_schema=XRayInput
        )

    # 5. Gateway Genérico
    elif mcp_tool.name == "call_metabase_api":
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description="API Genérica. Úsala para 'get_api_database' (listar DBs) o 'get_api_database_id_metadata' (ver esquema).",
            args_schema=MetabaseToolInput
        )

    # 6. Fallback para otras herramientas
    else:
        return StructuredTool.from_function(
            func=None, coroutine=wrapped_tool, name=mcp_tool.name,
            description=mcp_tool.description or "Herramienta Metabase"
        )

# --- 3. CLASE DEL AGENTE ---
class LangChainBrainAgent:
    def __init__(self, tools: list):
        brain_logger.info("Inicializando Brain Agent V2 (SQL Aware & Card Execution)...")
        self.llm = ChatOpenAI(
            base_url=settings.LM_STUDIO_URL, 
            api_key="not-required",
            model=settings.LLM_MODEL_NAME, 
            temperature=0
        )
        
        self.tools = tools
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        # Prompt actualizado con la nueva prioridad
        system_prompt = """Eres un Arquitecto de Datos Senior experto en Metabase y SQL.

        ### 🎯 TUS OBJETIVOS
        1. Responder preguntas de negocio extrayendo datos reales.
        2. Crear activos persistentes (Cards/Dashboards) cuando se solicite.
        3. Explorar la base de datos inteligentemente antes de consultar.

        ### 🛠️ TUS HERRAMIENTAS (Úsalas en este orden de prioridad)

        1️⃣ **PARA CONSULTAR DATOS DE UNA CARD EXISTENTE ("Ejecuta la card 280", "Ver reporte ventas"):**
           - Usa `execute_saved_card`.
           - Solo necesitas el ID de la card. Esta es la forma más segura si el usuario te da un ID.

        2️⃣ **PARA PREGUNTAS PUNTUALES ("¿Cuántas ventas...?", "¿Cuál fue el error...?"):**
           - Usa `run_sql_query`.
           - NO crees una card guardada a menos que el usuario lo pida explícitamente.
           - *Requiere:* `database_id` (búscalo primero si no lo sabes) y `sql_query`.

        3️⃣ **PARA CREAR CONTENIDO ("Crea una card para...", "Guarda esta consulta"):**
           - Usa `create_sql_card`.
           - Esto genera un enlace permanente.

        4️⃣ **PARA ANÁLISIS PROFUNDOS/DASHBOARDS ("Analiza la tabla X", "Dame un resumen visual"):**
           - Usa `generate_and_save_xray`.
           - Esto crea un Dashboard entero con múltiples gráficos.

        5️⃣ **PARA EXPLORAR (El paso cero):**
           - Usa `call_metabase_api` con `operation_id="get_api_database"` para ver qué DBs existen y sus IDs.
           - Usa `call_metabase_api` con `operation_id="get_api_database_id_metadata"` para ver tablas y columnas.

        ### 🧠 REGLAS DE ORO
        - **Siempre busca el `database_id` primero**: No inventes el ID. Si no lo sabes, lista las bases de datos.
        - **SQL Limpio**: Escribe SQL compatible con el motor (Postgres/MySQL). Siempre incluye `LIMIT 100` si no hay agregaciones.
        - **Errores**: Si `run_sql_query` falla, revisa el esquema de la tabla (nombres de columnas) e intenta de nuevo.
        - **Respuestas**: Si obtienes datos JSON, formatéalos como una tabla Markdown bonita o resume los hallazgos en texto claro.

        - Listar todas las Cards: `call_metabase_api(operation_id="get_api_card", payload={{}})`
        - Listar Dashboards: `call_metabase_api(operation_id="get_api_dashboard", payload={{"f": "all"}})`
        - Listar Tablas: `call_metabase_api(operation_id="get_api_database_id_metadata", payload={{"id": ...}})`

        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=10 
        )

    async def ainvoke(self, input_data: dict):
        try:
            res = await self.agent_executor.ainvoke(input_data)
            return {"output": res["output"]}
        except Exception as e:
            return {"output": f"❌ Error en el agente: {str(e)}"}

# --- FACTORY ---
async def build_brain_agent(session):
    brain_logger.info("Conectando con herramientas MCP...")
    mcp_tools = await session.list_tools()
    langchain_tools = [mcp_to_langchain_tool(t, session) for t in mcp_tools.tools]
    return LangChainBrainAgent(tools=langchain_tools)