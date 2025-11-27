import logging
import sys
from contextlib import AsyncExitStack
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Importaciones MCP y Agentes
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_core.messages import HumanMessage

# Tus módulos
from app.api.v1 import agents as agents_router
from app.agents import brain_agent as brain_agent_module # Metabase
from app.agents import mcp_agent as mcp_agent_module     # OpenWebUI
from app.agents.orchestrator import build_orchestrator   # Orquestador

# Configuración de Logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("PrismaFinanceAPI")

app = FastAPI(title="PrismaFinance API", version="2.0.0")
app.include_router(agents_router.router, prefix="/v1")

# --- URLS DE LOS SERVIDORES MCP ---
# m.py corriendo en 9002 (Metabase)
MCP_METABASE_URL = "http://127.0.0.1:9002/sse"
# p.py corriendo en 9001 (Open WebUI)
MCP_OPENWEBUI_URL = "http://127.0.0.1:9001/sse"

# Modelo para recibir peticiones
class UserQuery(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Sistema Multi-Agente...")
    app.state.exit_stack = AsyncExitStack()

    # --- 1. INICIAR AGENTE METABASE (Brain) ---
    # Ahora conectamos remotamente al puerto 9002 en lugar de usar Docker localmente
    try:
        logger.info(f"📊 Conectando a Metabase MCP ({MCP_METABASE_URL})...")
        # Conexión SSE
        streams_mb = await app.state.exit_stack.enter_async_context(sse_client(MCP_METABASE_URL))
        session_mb = await app.state.exit_stack.enter_async_context(ClientSession(streams_mb[0], streams_mb[1]))
        await session_mb.initialize()
        
        # Construir el agente usando la nueva factoría en brain_agent.py
        brain_agent_instance = await brain_agent_module.build_brain_agent(session_mb)
        logger.info("✅ Brain Agent listo.")
    except Exception as e:
        logger.error(f"⚠️ Fallo al conectar con Metabase MCP (m.py): {e}")
        brain_agent_instance = None

    # --- 2. INICIAR AGENTE OPEN WEBUI (MCP Worker) ---
    # Conexión remota al puerto 9001
    try:
        logger.info(f"🔧 Conectando a Open WebUI ({MCP_OPENWEBUI_URL})...")
        streams_ow = await app.state.exit_stack.enter_async_context(sse_client(MCP_OPENWEBUI_URL))
        session_ow = await app.state.exit_stack.enter_async_context(ClientSession(streams_ow[0], streams_ow[1]))
        await session_ow.initialize()
        
        # Construir el agente
        mcp_agent_instance = await mcp_agent_module.build_mcp_worker_agent(session_ow)
        logger.info("✅ MCP OpenWebUI Agent listo.")
    except Exception as e:
        logger.error(f"⚠️ Fallo al conectar con Open WebUI MCP (p.py): {e}")
        mcp_agent_instance = None

    # --- 3. PREPARAR EL ORQUESTADOR ---
    try:
        app.state.orchestrator = build_orchestrator()
        app.state.agents_config = {
            "brain_agent": brain_agent_instance,
            "mcp_agent": mcp_agent_instance
        }
        logger.info("🤖 ORQUESTADOR OPERATIVO.")
    except Exception as e:
        logger.error(f"❌ Error fatal configurando orquestador: {e}")
        raise RuntimeError("Startup failed") from e

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "exit_stack"):
        await app.state.exit_stack.aclose()
    logger.info("🔌 API Apagada.")

# --- ENDPOINT PRINCIPAL ---
@app.post("/chat", tags=["Orchestrator"])
async def chat_endpoint(request: UserQuery, fastapi_req: Request):
    """Endpoint único que recibe la pregunta y el orquestador decide."""
    
    orchestrator = fastapi_req.app.state.orchestrator
    config = fastapi_req.app.state.agents_config
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    
    # Ejecutar el grafo del orquestador
    result = await orchestrator.ainvoke(inputs, config={"configurable": config})
    
    return {
        "response": result["final_response"],
        "routed_to": result.get("next_agent", "unknown")
    }