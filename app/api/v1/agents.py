import time
import re
import json
import logging
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional
from langchain_core.messages import HumanMessage

# Importamos utilidades del brain_agent para el endpoint SQL directo
from app.agents.brain_agent import normalize_sql

router = APIRouter()
logger = logging.getLogger("AgentsAPI")

# --- Modelos de Datos (Igual que antes) ---
class AgentInvocationRequest(BaseModel):
    prompt: str

class AgentResponse(BaseModel):
    original_query: str
    agent_output: Any
    processing_time_seconds: float

class Model(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "prisma-finance"

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

# --- Modelos Compatibles con OpenAI ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class OpenAIChatRequest(BaseModel):
    model: str = "orchestrator"
    messages: List[ChatMessage]

class ChatResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None

class Choice(BaseModel):
    message: ChatResponseMessage
    finish_reason: Literal["stop", "tool_calls"] = "stop"
    index: int = 0

class OpenAIChatResponse(BaseModel):
    id: str = "chatcmpl-prisma-finance"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]


# --- Endpoints ---

def ensure_select_only(sql: str):
    """Raise an exception if the SQL is not a SELECT statement."""
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|GRANT|REVOKE|CALL|EXEC)\b",
                 sql, flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Only SELECT (read-only) queries are allowed.")

@router.get("/chat/sql", tags=["Agents"], response_model=AgentResponse)
async def run_sql_direct(
    request: Request,
    q: str = Query(..., description="SQL SELECT query to execute"),
    database_id: int = Query(..., description="ID of the Metabase database"),
    limit: int = Query(200, ge=1, le=10000, description="Max rows to return"),
):
    """
    Ejecuta SQL directamente usando el Brain Agent (Metabase).
    """
    start_time = time.time()
    sql = normalize_sql(q)
    ensure_select_only(sql)

    # CORRECCIÓN 1: Acceder al agente a través del diccionario de configuración
    agents_config = getattr(request.app.state, "agents_config", {})
    brain_agent = agents_config.get("brain_agent")

    if not brain_agent:
        raise HTTPException(status_code=503, detail="Brain Agent (Metabase) no está inicializado o disponible.")

    # Ejecutar consulta
    try:
        result = await brain_agent.ainvoke_sql_direct(database_id, sql, limit)
    except Exception as e:
        logger.error(f"Error en run_sql_direct: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AgentResponse(
        original_query=q,
        agent_output=result,
        processing_time_seconds=round(time.time() - start_time, 2)
    )

@router.get("/models", tags=["Models"], response_model=ModelList)
async def list_models():
    """Lists the available models/agents that can be invoked."""
    model_data = [
        Model(id="orchestrator-v2", owned_by="system"),
        Model(id="gpt-3.5-turbo", owned_by="openai"), # Compatibilidad
        Model(id="gpt-4", owned_by="openai"),         # Compatibilidad
    ]
    return ModelList(data=model_data)

@router.post("/chat/completions", tags=["Chat"], response_model=OpenAIChatResponse)
async def openai_chat_endpoint(fastapi_request: Request, request: OpenAIChatRequest):
    """
    Endpoint compatible con OpenAI que ahora usa el ORQUESTADOR.
    """
    # Validar entrada
    if not request.messages or not request.messages[-1].content:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    user_query = request.messages[-1].content
    
    # CORRECCIÓN 2: Usar el Orquestador guardado en el estado
    orchestrator = getattr(fastapi_request.app.state, "orchestrator", None)
    agents_config = getattr(fastapi_request.app.state, "agents_config", {})

    if not orchestrator:
        # Fallback de emergencia si el orquestador falló al inicio
        raise HTTPException(status_code=503, detail="El sistema de orquestación no está listo.")

    try:
        # Invocar al Grafo del Orquestador
        # Pasamos 'agents_config' para que los nodos sepan a quién llamar
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        result = await orchestrator.ainvoke(
            inputs, 
            config={"configurable": agents_config}
        )
        
        # Extraer respuesta final del estado del grafo
        final_response = result.get("final_response", "No response generated.")
        
    except Exception as e:
        logger.error(f"Error ejecutando orquestador: {e}", exc_info=True)
        final_response = f"Error interno del sistema: {str(e)}"

    # Formatear respuesta estilo OpenAI
    response_message = ChatResponseMessage(content=str(final_response))
    choice = Choice(message=response_message)
    
    return OpenAIChatResponse(model=request.model, choices=[choice])