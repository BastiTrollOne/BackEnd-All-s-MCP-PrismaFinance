import logging
import re
import json
from typing import List, Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field # Importante para definir esquemas

# Ajusta estos imports a tu estructura de carpetas real
from app.core.config import settings 
# from app.agents.sql_agent import create_sql_agent_chain (Si lo usas)

# --- FUNCIONES AUXILIARES RESTAURADAS (Necesarias para otros módulos) ---
def normalize_sql(sql: str) -> str:
    """Limpia espacios extra y puntos y coma finales en SQL."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";" ))

def enforce_limit(sql: str, default_limit: int = 200) -> str:
    """Asegura que la consulta tenga un límite para no saturar."""
    if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {default_limit}"

brain_logger = logging.getLogger("LangChainBrainAgent")

# --- 1. ESQUEMAS DE ENTRADA (Input Schemas) ---
# Esto ayuda al LLM a entender qué argumentos son obligatorios
class MetabaseToolInput(BaseModel):
    operation_id: str = Field(..., description="El ID técnico de la operación (ej: 'get_api_database')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Diccionario con los parámetros (query, body, path). Ej: {'include': 'tables'}")

# --- 2. ADAPTADOR DE HERRAMIENTAS MEJORADO ---
# --- 2. ADAPTADOR DE HERRAMIENTAS MEJORADO (CORREGIDO) ---
def mcp_to_langchain_tool(mcp_tool, session):
    
    # CASO A: Herramienta Gateway Compleja (call_metabase_api)
    if "call_metabase_api" in mcp_tool.name:
        async def wrapped_gateway(operation_id: str, payload: Dict[str, Any] = {}):
            brain_logger.info(f"🛠️ Gateway: {operation_id} | Payload keys: {list(payload.keys())}")
            try:
                result = await session.call_tool(mcp_tool.name, arguments={"operation_id": operation_id, "payload": payload})
                if hasattr(result, 'content') and result.content:
                    return next((c.text for c in result.content if hasattr(c, 'text')), str(result))
                return str(result)
            except Exception as e:
                return f"Error Gateway: {str(e)}"

        return StructuredTool.from_function(
            func=None,
            coroutine=wrapped_gateway,
            name=mcp_tool.name,
            description="Ejecuta una operación en Metabase. Requiere 'operation_id' (str) y 'payload' (dict).",
            args_schema=MetabaseToolInput 
        )

    # CASO B: Herramientas Nativas/Simples (Como generate_and_save_xray)
    else:
        async def wrapped_generic(**kwargs):
            brain_logger.info(f"🛠️ Tool: {mcp_tool.name} | Args: {kwargs}")
            try:
                # Pasamos los argumentos tal cual vienen del LLM
                result = await session.call_tool(mcp_tool.name, arguments=kwargs)
                if hasattr(result, 'content') and result.content:
                    return next((c.text for c in result.content if hasattr(c, 'text')), str(result))
                return str(result)
            except Exception as e:
                return f"Error Tool: {str(e)}"

        # Definimos un esquema específico para tu nueva herramienta
        # para que el LLM sepa qué argumentos enviar.
        if mcp_tool.name == "generate_and_save_xray":
            class XRayInput(BaseModel):
                table_id: int = Field(..., description="El ID numérico de la tabla para generar el dashboard.")
            
            return StructuredTool.from_function(
                func=None,
                coroutine=wrapped_generic,
                name=mcp_tool.name,
                description=mcp_tool.description or "Genera y guarda un dashboard automático.",
                args_schema=XRayInput
            )

        # Fallback para otras herramientas (get_operation_details, etc.)
        return StructuredTool.from_function(
            func=None,
            coroutine=wrapped_generic,
            name=mcp_tool.name,
            description=mcp_tool.description or "Herramienta Metabase"
        )

# --- 3. FUNCIONES AUXILIARES ---
def format_markdown_table(data: Any) -> str:
    """Intenta convertir una lista de diccionarios en una tabla Markdown"""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        # Si Metabase devuelve {data: [...], total: ...}
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list) or not data:
            return str(data)

        # Tomar cabeceras del primer elemento
        if not isinstance(data[0], dict):
            return str(data)
            
        headers = list(data[0].keys())
        # Filtrar cabeceras muy largas o técnicas si es necesario
        headers = [h for h in headers if not h.startswith("_")] 
        
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in data[:20]: # Limitar a 20 filas para no saturar
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
            
        if len(data) > 20:
            lines.append(f"\n*... y {len(data) - 20} filas más.*")
            
        return "\n".join(lines)
    except Exception:
        return str(data)

# --- 4. CLASE DEL AGENTE ---
class LangChainBrainAgent:
    def __init__(self, tools: list):
        brain_logger.info("Inicializando Brain Agent...")
        self.llm = ChatOpenAI(
            base_url=settings.LM_STUDIO_URL, 
            api_key="not-required",
            model=settings.LLM_MODEL_NAME, 
            temperature=0
        )
        
        self.tools = tools
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        # Prompt ACTUALIZADO para usar la nueva herramienta "generate_and_save_xray"
        system_prompt = """Eres un Analista de Datos experto en Metabase.
        
        ⚠️ **REGLAS CRÍTICAS DE EJECUCIÓN:**
        1. **HERRAMIENTAS DISPONIBLES:** Tienes dos tipos de herramientas principales:
           - `call_metabase_api`: Tu navaja suiza para todo (consultar, listar, ver esquemas). Requiere `operation_id` y `payload`.
           - `generate_and_save_xray`: Una herramienta ESPECIALIZADA y AUTOMÁTICA para crear dashboards.

        2. **NO INVENTES DATOS:** Si una herramienta falla, reporta el error.

        ### 🛠️ EJEMPLOS DE `call_metabase_api` (Usa llaves dobles {{ }}):
        
        - **Listar bases de datos:**
          `call_metabase_api(operation_id="get_api_database", payload={{"include": "tables"}})`
          
        - **Obtener esquema/tablas (ID=1):**
          `call_metabase_api(operation_id="get_api_database_id_metadata", payload={{"id": 1}})`

        ### 🧠 TUS ESTRATEGIAS MAESTRAS:

        1. **GENERACIÓN DE DASHBOARDS AUTOMÁTICOS (La Vía Rápida):**
           - Esta es tu mejor habilidad. Si te piden un "Automagic Dashboard" o "X-Ray":
           - 🟢 Paso 1: Busca el ID de la tabla de interés usando `call_metabase_api` con `get_api_database_id_metadata`.
           - 🟢 Paso 2: **NO** intentes generar el JSON manualmente.
           - 🟢 Paso 3: Llama DIRECTAMENTE a la herramienta `generate_and_save_xray(table_id=ID_ENCONTRADO)`.
             - Esta herramienta genera el diseño y lo guarda en un solo paso.
             - Devuelve la URL y el ID del dashboard creado.

        2. **EXPLORACIÓN Y CONSULTAS:**
           - Usa siempre `call_metabase_api` para `post_api_dataset` (queries) o `get_api_database`.

        3. **MANEJO DE ERRORES:**
           - Si te piden una tabla de datos, devuelve el JSON crudo; yo lo formatearé.
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
            handle_parsing_errors=True
        )

    async def ainvoke(self, input_data: dict):
        try:
            res = await self.agent_executor.ainvoke(input_data)
            # Post-procesamiento para intentar formatear tablas automáticamente
            final_output = res["output"]
            if "data" in str(final_output) or "[" in str(final_output):
                 # Intento simple de detectar si hay datos crudos que formatear
                 pass 
            return {"output": final_output}
        except Exception as e:
            return {"output": f"❌ Error fatal en el agente: {str(e)}"}

# --- FACTORY ---
async def build_brain_agent(session):
    brain_logger.info("Conectando con herramientas MCP...")
    mcp_tools = await session.list_tools()
    # Convertimos las herramientas usando el nuevo adaptador con esquema estricto
    langchain_tools = [mcp_to_langchain_tool(t, session) for t in mcp_tools.tools]
    return LangChainBrainAgent(tools=langchain_tools)