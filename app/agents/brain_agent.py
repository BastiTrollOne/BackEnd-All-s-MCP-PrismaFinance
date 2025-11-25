import logging
import re
import json
import os
from typing import List, Any, Tuple, Dict, Optional

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic import hub
from langchain.tools import tool
from fastapi import HTTPException
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings
from app.agents.sql_agent import create_sql_agent_chain

brain_logger = logging.getLogger("LangChainBrainAgent")

# --- MCP Client Setup ---
mcp_client: MultiServerMCPClient = None
mcp_tools_cache: Dict[str, Any] = {}  # Global cache for tools


def initialize_mcp_client() -> MultiServerMCPClient:
    """Initializes the MCP client using direct environment variable passing for Docker."""
    global mcp_client
    mcp_client = MultiServerMCPClient({
        "MCP_METABASE": {
            "command": "docker",
            "args": [
                "run", "-i", "--rm",
                # This flag ensures that 'host.docker.internal' correctly resolves to the host machine across different platforms (Windows, Mac, Linux).
                "--add-host=host.docker.internal:host-gateway",
                "-e", f"METABASE_URL={settings.METABASE_URL_FOR_DOCKER}",
                "-e", f"METABASE_USERNAME={settings.METABASE_USERNAME}",
                "-e", f"METABASE_PASSWORD={settings.METABASE_PASSWORD}",
                "mcp/metabase"
            ],
            "transport": "stdio",
            # This env block seems critical for the mcp/metabase container on Windows
            "env": {
                "LOCALAPPDATA": os.environ.get("LOCALAPPDATA"),
                "ProgramData": os.environ.get("ProgramData"),
                "ProgramFiles": os.environ.get("ProgramFiles")
            }
        }
    })
    return mcp_client


# --- Helper Functions ---
def normalize_sql(sql: str) -> str:
    """Removes extra whitespace and trailing semicolons from SQL."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";" ))


def enforce_limit(sql: str, default_limit: int = 200) -> str:
    """Adds a LIMIT clause to a SQL query if it doesn't have one."""
    return sql if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE) else f"{sql} LIMIT {default_limit}"


def format_markdown_table(rows: List[Dict[str, Any]], max_rows: int = 20) -> str:
    """
    Convierte una lista de dicts en una tabla Markdown.
    Limita el número de filas mostradas.
    """
    if not rows:
        return "La consulta no devolvió filas."

    # Usamos las keys del primer dict como cabeceras
    headers = list(rows[0].keys())

    # Cabecera de la tabla
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Filas (limitadas)
    for row in rows[:max_rows]:
        line = "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
        lines.append(line)

    if len(rows) > max_rows:
        lines.append(f"\n_Mostrando las primeras {max_rows} filas de {len(rows)}._")

    return "\n".join(lines)


def convert_output_to_markdown(output: Any, max_rows: int = 20) -> str:
    """
    Convierte cualquier tipo de salida (JSON, dict, list, etc.) en Markdown legible.
    - Si es una tabla (lista de dicts) → tabla Markdown
    - Si es dict/list normal → lista Markdown bonita
    - Si es texto → se devuelve igual
    """

    # Si ya es string y no es JSON válida, devolver tal cual
    if isinstance(output, str):
        try:
            output_json = json.loads(output)
            output = output_json
        except Exception:
            return output  # texto normal, devolver sin tocar

    # LISTA DE FILAS (TABLA)
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return format_markdown_table(output, max_rows=max_rows)

    # DICT → convertir a bloques Markdown
    if isinstance(output, dict):
        lines = []
        for k, v in output.items():
            lines.append(f"**{k}:** {v}")
        return "\n".join(lines)

    # Otras listas
    if isinstance(output, list):
        return "\n".join(f"- {item}" for item in output)

    # Fallback
    return str(output)


# --- Tools ---


async def get_default_database(agent_instance):
    """
    Detecta automáticamente la base de datos por defecto usando la herramienta list_databases.
    - Prefiere bases con engine != 'h2'
    - No requiere configuración manual en el backend.
    """
    brain_logger.info("Finding default database using 'list_databases' tool...")
    raw = await agent_instance.list_db_tool.ainvoke({})
    brain_logger.debug(f"Raw output from list_databases: {repr(raw)}")

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            brain_logger.error("list_databases returned plain string; cannot determine databases.")
            raise ValueError("Invalid response from 'list_databases': plain string without JSON.")

    # Normalizar a lista
    if isinstance(raw, list):
        dbs = raw
    elif isinstance(raw, dict):
        for key in ("databases", "data", "results", "items"):
            if key in raw and isinstance(raw[key], list):
                dbs = raw[key]
                break
        else:
            dbs = [raw]
    else:
        brain_logger.error(f"Unexpected type from list_databases: {type(raw)}")
        raise ValueError("Unexpected 'list_databases' result format.")

    dbs = [d for d in dbs if isinstance(d, dict)]
    if not dbs:
        brain_logger.error(f"No valid database entries found in: {repr(raw)}")
        raise ValueError("No valid databases found in 'list_databases' response.")

    non_h2 = [
        d for d in dbs
        if str(d.get("engine", "")).lower() != "h2"
        and d.get("id") not in (None, "")
    ]

    chosen = (non_h2 or dbs)[0]

    if "id" not in chosen or chosen["id"] in (None, ""):
        brain_logger.error(f"Chosen DB without valid id: {chosen}")
        raise ValueError("Chosen database has no valid 'id' field.")

    db_id = chosen["id"]
    db_name = chosen.get("name", "<unnamed>")
    brain_logger.info(f"Selected default database '{db_name}' (ID={db_id}) automatically.")
    return db_id, db_name


def looks_like_sql(text: str) -> bool:
    return bool(re.match(
        r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
        text,
        re.IGNORECASE
    ))


@tool
async def execute_sql(
    user_question: Optional[str] = None,
    query: Optional[str] = None
) -> str:
    """
    Executes a SQL query or answers a data question.
    - Si recibe SQL directo, lo ejecuta tal cual contra Metabase (MCP).
    - Si recibe lenguaje natural, genera SQL con sql_agent_chain y luego lo ejecuta.
    - La base de datos se selecciona automáticamente.
    """
    input_for_sql = user_question or query

    if not input_for_sql:
        last = mcp_tools_cache.get("last_user_input")
        if isinstance(last, str):
            input_for_sql = last.strip()

    if not input_for_sql:
        return "Error: This tool requires either a user_question or a query to execute."

    brain_logger.info(f"Smart tool 'execute_sql' invoked with: '{input_for_sql}'")

    agent_instance = mcp_tools_cache.get("agent_instance")
    if not agent_instance:
        raise ValueError("Could not access the parent agent instance.")

    # 1) DB automática (sin config manual)
    database_id, db_name = await get_default_database(agent_instance)

    # 2) Decidir si es SQL o lenguaje natural
    if looks_like_sql(input_for_sql):
        # Es SQL → NO usamos LLM para inventar nada
        final_sql = input_for_sql
        brain_logger.info("Input detected as raw SQL. Skipping sql_agent_chain.")
    else:
        # Es NL → usamos el sql_agent_chain SOLO para generar SQL
        brain_logger.info("Input detected as natural language. Using sql_agent_chain to generate SQL...")
        generated = await agent_instance.sql_agent_chain.ainvoke({
            "question": input_for_sql,
            "database_id": database_id,
        })

        # El sql_agent_chain puede devolver string o dict
        if isinstance(generated, dict):
            final_sql = generated.get("sql") or generated.get("query") or str(generated)
        else:
            final_sql = str(generated)

        brain_logger.info(f"SQL generated by sql_agent_chain: {final_sql}")

    # 3) Ejecutar SIEMPRE vía MCP (aquí es donde se consulta de verdad)
    result = await agent_instance.ainvoke_sql_direct(
        database_id=database_id,
        raw_sql=final_sql
    )

    # --- Formatear salida limpia si el resultado viene del MCP ---

    # Si viene como string JSON, lo parseamos
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            # No era JSON, devolvemos texto tal cual (ya no JSON crudo)
            return result

    # Si tiene estructura Metabase (data -> rows/cols), simplificamos a tabla
    if isinstance(result, dict):
        data = result.get("data") or {}
        rows = data.get("rows") or []
        cols_meta = data.get("cols") or []

        if rows and cols_meta:
            cols = [
                (c.get("name") or c.get("display_name"))
                for c in cols_meta
                if isinstance(c, dict)
            ]
            simplified = [dict(zip(cols, r)) for r in rows]

            # ✅ Devolver tabla Markdown en vez de JSON crudo
            return format_markdown_table(simplified, max_rows=20)

    # Fallback: pasar por el conversor genérico a Markdown
    return convert_output_to_markdown(result)


@tool
async def list_products(
    user_question: Optional[str] = None,
    query: Optional[str] = None
) -> str:
    """
    Alias de compatibilidad. Si el modelo intenta llamar 'list_products',
    delegamos en execute_sql para que consulte la base de datos real.
    """
    return await execute_sql.ainvoke({
        "user_question": user_question,
        "query": query
    })

@tool
async def list_databases_pretty() -> str:
    """
    Lista las bases de datos disponibles en Metabase en forma de tabla Markdown.
    Muestra solo campos relevantes (id, name, engine, is_sample).
    """
    agent_instance = mcp_tools_cache.get("agent_instance")
    if not agent_instance:
        raise ValueError("Could not access the parent agent instance.")

    brain_logger.info("Listing databases using 'list_databases' tool...")

    raw = await agent_instance.list_db_tool.ainvoke({})
    brain_logger.debug(f"Raw output from list_databases_pretty: {repr(raw)}")

    # Normalizar igual que en get_default_database
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            brain_logger.error("list_databases returned plain string; cannot determine databases.")
            return f"No se pudo interpretar la respuesta de list_databases: {raw}"

    if isinstance(raw, list):
        dbs = raw
    elif isinstance(raw, dict):
        for key in ("databases", "data", "results", "items"):
            if key in raw and isinstance(raw[key], list):
                dbs = raw[key]
                break
        else:
            dbs = [raw]
    else:
        brain_logger.error(f"Unexpected type from list_databases_pretty: {type(raw)}")
        return "Formato inesperado en la respuesta de list_databases."

    dbs = [d for d in dbs if isinstance(d, dict)]
    if not dbs:
        return "No se encontraron bases de datos."

    # Nos quedamos solo con campos interesantes
    rows = []
    for d in dbs:
        rows.append({
            "id": d.get("id"),
            "name": d.get("name"),
            "engine": d.get("engine"),
            "is_sample": d.get("is_sample"),
        })

    return format_markdown_table(rows, max_rows=50)


# --- LangChainBrainAgent Class ---
class LangChainBrainAgent:
    """
    Agent Cerebro: Uses an LLM and Metabase tools to answer data-related questions.
    It decides automatically whether to respond directly or use a tool.
    """
    def __init__(self, tools: list):
        brain_logger.info("Initializing Brain Agent (LangChain with AgentExecutor)...")
        self.llm = ChatOpenAI(
            base_url=settings.LM_STUDIO_URL,
            api_key="not-required",
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            top_p=0.1,
            max_tokens=2048,  # Reduce max tokens to prevent server overload
            timeout=300  # Set a 5-minute timeout to allow the local LLM to process large prompts
        )

        # We keep the original tools to pass to the SQL chain, which needs get_database_schema
        original_tools = tools

        # But the agent itself will only see the high-level tools to avoid confusion.
        # We remove low-level tools that are now encapsulated in our smart tool.
        self.tools = [
            t for t in tools if t.name not in ["execute_query", "get_database_schema"]
                      ] + [execute_sql, list_products, list_databases_pretty]

        self.sql_agent_chain = create_sql_agent_chain(original_tools)
        # The mcp_tool is the original 'execute_query' tool, needed for ainvoke_sql_direct
        self.mcp_client = mcp_client  # Store client instance for direct calls
        self.mcp_tool = next((t for t in original_tools if t.name == "execute_query"), None)
        if not self.mcp_tool:
            brain_logger.error("Critical: 'execute_query' tool not found.")
        self.list_db_tool = next((t for t in original_tools if t.name == "list_databases"), None)
        if not self.list_db_tool:
            brain_logger.error("Critical: 'list_databases' tool not found.")

        # We directly define the prompt in Spanish to ensure consistent behavior,
        # instead of relying on a fallback from a potentially English hub prompt.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
Eres un Agente de Inteligencia Artificial experto en Metabase, conectado mediante MCP (Model Context Protocol).
Tu misión es ayudar al usuario a obtener datos y respuestas precisas utilizando EXCLUSIVAMENTE las herramientas disponibles.

TUS HERRAMIENTAS DISPONIBLES:
{{tool_names}}

INSTRUCCIONES MAESTRAS (Síguelas estrictamente):

1. USO GENERAL DE HERRAMIENTAS
- NO inventes datos. NO adivines respuestas.
- Para cualquier pregunta que requiera datos reales (por ejemplo: ventas, productos, usuarios, métricas, etc.), SIEMPRE debes llamar a una herramienta.
- NUNCA inventes nombres de herramientas. Usa solo las listadas arriba en {{tool_names}}.

2. ENRUTAMIENTO SEGÚN EL TIPO DE PREGUNTA

2.1. Consultas de datos (tablas, agregaciones, “productos más vendidos”, “ventas por mes”, etc.)
- Si la pregunta del usuario es sobre métricas, listados, agregaciones o análisis de datos (por ejemplo:
  - "productos más vendidos"
  - "ventas por categoría"
  - "top 10 clientes"
  ), DEBES usar herramientas de consulta de datos, típicamente:
  - `execute_sql` (consulta directa o generación de SQL a partir de lenguaje natural)
  - o `execute_card` si el usuario menciona explícitamente una Card ya existente.
- NO uses `list_databases` ni herramientas de metadata solo para responder preguntas de negocio.
- Ejemplo:
  Usuario: "Muéstrame los productos más vendidos"
  → Herramienta adecuada: `execute_sql` para obtener una tabla con productos y total_vendido.

2.2. Dashboards y Cards (crear, actualizar, agregar tarjetas, etc.)
- Si la pregunta menciona explícitamente dashboards, paneles, tarjetas o gráficos guardados, prioriza:
  - `create_card`, `update_card`, `delete_card`, `execute_card`, `list_cards`
  - `create_dashboard`, `update_dashboard`, `delete_dashboard`, `list_dashboards`
  - `add_card_to_dashboard`, `remove_card_from_dashboard`, `update_dashboard_card`, `get_dashboard_cards`
- Flujo típico para "crea un dashboard de productos más vendidos":
  1) Usa `execute_sql` para determinar/generar el SQL correcto (si es necesario).
  2) Usa `create_card` con ese SQL para crear una Card.
  3) Usa `create_dashboard` para crear el dashboard (si el usuario lo pide).
  4) Usa `add_card_to_dashboard` para añadir la Card al dashboard.
- NO uses `list_databases` para este tipo de consultas. No tiene sentido listar bases de datos para "productos más vendidos".

2.3. Metadata y administración de bases de datos
- Usa `list_databases` (o una herramienta equivalente para listar bases) SOLAMENTE cuando el usuario pregunte explícitamente algo como:
  - "¿Qué bases de datos tengo?"
  - "Listame las bases de datos"
  - "Qué conexiones de base de datos hay?"
- Para inspeccionar tablas, esquemas o sincronización usa:
  - `get_database_tables`, `get_database_schema`, `sync_database_schema`, `get_database_sync_status`, etc.
- NO uses estas herramientas para responder preguntas de negocio (ventas, productos, usuarios, etc.), solo para información técnica o de administración.

2.4. Búsqueda de contenido y colecciones
- Si el usuario pide localizar dashboards, cards o contenidos por nombre o texto, puedes usar:
  - `search_content`, `list_collections`, `list_dashboards`, `list_cards`.

3. CONSULTAS SOBRE CAPACIDADES
- Si el usuario pregunta "¿Qué puedes hacer?", "Ayuda" o "¿Qué herramientas tienes?":
  - NO ejecutes ninguna herramienta.
  - Responde explicando tus capacidades principales en lenguaje natural y sencillo.
  - DESPUÉS de la explicación, crea una tabla Markdown con dos columnas: "Herramienta" y "Descripción".
    - En la columna "Herramienta", lista los nombres técnicos de tus herramientas (los obtienes de {{tool_names}}).
    - En la columna "Descripción", añade un resumen breve de lo que hace cada una.

4. FORMATO DE RESPUESTA
- Sé conciso y directo.
- Si necesitas aclaración, pídela ANTES de llamar a una herramienta.
- Cuando la respuesta provenga de datos tabulares (por ejemplo, resultados de `execute_sql` o `execute_card`),
  devuélvelos en formato de tabla Markdown (no devuelvas JSON crudo).
- AL FINAL DE CADA MENSAJE, incluye un pie de página indicando qué herramientas se utilizaron o "Ninguna" si fue una conversación general.

ESTRUCTURA OBLIGATORIA DEL PIE DE PÁGINA:
---
🛠 **Herramientas/Secuencia usada:** [Nombre_Herramienta_1] > [Nombre_Herramienta_2] (o "Ninguna")

EJEMPLOS DE COMPORTAMIENTO:

Ejemplo 1 — Consulta de datos:
Usuario: "¿Cuáles fueron las ventas de ayer?"
Asistente:
1) [Llamada a tool: execute_sql con una consulta adecuada para ventas de ayer]
2) Presenta el resultado en una tabla Markdown.
---
🛠 **Herramientas/Secuencia usada:** execute_sql

Ejemplo 2 — Capacidades:
Usuario: "¿Qué puedes hacer?"
Asistente:
- Explica en lenguaje natural que puedes ejecutar consultas, crear dashboards, listar cards, etc.
- Luego muestra una tabla con los nombres y descripciones de las herramientas de {{tool_names}}.
---
🛠 **Herramientas/Secuencia usada:** Ninguna

Ejemplo 3 — Dashboard de productos más vendidos:
Usuario: "Crea un dashboard de productos más vendidos"
Asistente:
1) Usa `execute_sql` (si es necesario) para determinar el SQL correcto (top productos ordenados por total vendido).
2) Usa `create_card` para crear una Card con ese SQL.
3) Usa `create_dashboard` para crear un nuevo dashboard (si el usuario lo pide explícitamente).
4) Usa `add_card_to_dashboard` para añadir la Card al dashboard.
Luego responde algo como:
"He creado el dashboard 'Productos más vendidos' con una card que muestra los productos con mayor total de ventas."
---
🛠 **Herramientas/Secuencia usada:** execute_sql > create_card > create_dashboard > add_card_to_dashboard

Ejemplo 4 — Listar bases de datos:
Usuario: "Listame las bases de datos disponibles"
Asistente:
1) Usa `list_databases` (o tu herramienta específica para listar bases).
2) Muestra el resultado en una tabla Markdown con columnas como id, nombre, engine, is_sample.
---
🛠 **Herramientas/Secuencia usada:** list_databases
"""
    ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # This global cache is a workaround to give the @tool access to the agent instance.
        global mcp_tools_cache
        mcp_tools_cache = {"agent_instance": self}

        brain_logger.info("Spanish prompt for Brain Agent loaded.")

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # Set to False to reduce terminal noise
            # Use a lambda to log the error and provide a user-friendly message.
            handle_parsing_errors=lambda e: (
                brain_logger.error(f"LLM output parsing error: {e}") or
                {"output": "I had trouble understanding the last step, please try rephrasing your request."}
            ),
            max_iterations=1,  # The agent's only job is to call one tool.
            early_stopping_method="force",
            return_intermediate_steps=True  # We want the raw output of the tool.
        )
        brain_logger.info("Brain Agent ready.")

    async def ainvoke(self, input_data: dict):
        user_query = input_data.get('input', 'Hello')
        brain_logger.info(f"Received query: '{user_query[:50]}...'")

        global mcp_tools_cache
        mcp_tools_cache["last_user_input"] = user_query

        try:
            # Dejamos que el agent executor decida si usar tools o no
            brain_logger.info("Invoking agent executor...")
            agent_decision = await self.agent_executor.ainvoke(input_data)
            intermediate_steps = agent_decision.get("intermediate_steps", [])

            if intermediate_steps:
                # ✅ Hubo tool → los datos vienen de una herramienta (Metabase, etc.)
                action, observation = intermediate_steps[0]
                brain_logger.info(f"Agent used tool '{action.tool}'. Returning observation.")
                return {"output": convert_output_to_markdown(observation)}
            else:
                # ❌ Sin tool → El agente respondió directamente (conversación general, etc.)
                # Devolvemos la respuesta generada por el LLM, que está en la clave 'output'.
                brain_logger.info("Agent answered directly without using tools.")
                return {"output": agent_decision.get("output", "No se generó una respuesta.")}

        except Exception as e:
            brain_logger.error(f"Error in AgentExecutor: {e}", exc_info=True)
            return {"output": f"Error processing the request: {e}"}

    async def run_tool_calling_agent(self, input_data: dict):
        """The original agent executor logic for calling non-SQL tools."""
        brain_logger.debug("Executing generic tool-calling agent...")
        response_dict = await self.agent_executor.ainvoke(input_data)
        intermediate_steps = response_dict.get("intermediate_steps", [])
        if intermediate_steps:
            action, observation = intermediate_steps[0]
            brain_logger.info(f"Tool '{action.tool}' returned: {observation}")
            return {"output": convert_output_to_markdown(observation)}
        return {"output": "El agente no llamó a ninguna herramienta."}

    async def ainvoke_sql_direct(self, database_id: int, raw_sql: str, limit: int = 200):
        if not self.mcp_tool:
            error_msg = "'execute_query' tool not initialized. The agent cannot process direct SQL queries."
            brain_logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        sql = normalize_sql(raw_sql)
        sql = enforce_limit(sql, limit)
        brain_logger.info(f"Calling 'execute_query' tool directly on DB {database_id}")
        brain_logger.info(f"SQL: {sql}")

        try:
            result = await self.mcp_tool.ainvoke({"database_id": database_id, "query": sql})
            # Log en JSON, pero lo que se devuelve hacia arriba ya pasa por formatters
            brain_logger.info(f"Query result (raw): {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        except Exception as e:
            brain_logger.error(f"Error in direct tool call: {e}", exc_info=True)
            # Re-raise as an HTTPException to be handled by FastAPI
            raise HTTPException(status_code=400, detail=f"Error executing SQL: {e}")
