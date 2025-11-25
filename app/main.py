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
from app.agents import mcp_agent as mcp_agent_module     # OpenWebUI (El archivo que creamos antes)
from app.agents.orchestrator import build_orchestrator   # El nuevo orquestador

# Configuración de Logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("PrismaFinanceAPI")

app = FastAPI(title="PrismaFinance API", version="2.0.0")
app.include_router(agents_router.router, prefix="/v1")

# URL del servidor p.py (Open WebUI)
MCP_OPENWEBUI_URL = "http://localhost:9001/sse"

# Modelo para recibir peticiones
class UserQuery(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Sistema Multi-Agente...")
    app.state.exit_stack = AsyncExitStack()

    try:
        # --- 1. INICIAR AGENTE METABASE (Brain) ---
        logger.info("📊 Conectando a Metabase...")
        # Esto arrancará el contenedor Docker de Metabase si usas initialize_mcp_client
        brain_agent_module.initialize_mcp_client() 
        metabase_tools = await brain_agent_module.mcp_client.get_tools(server_name="MCP_METABASE")
        brain_agent_instance = brain_agent_module.LangChainBrainAgent(tools=metabase_tools)
        logger.info(f"✅ Brain Agent listo con {len(metabase_tools)} herramientas.")

        # --- 2. INICIAR AGENTE OPEN WEBUI (MCP Worker) ---
        logger.info(f"🔧 Conectando a Open WebUI ({MCP_OPENWEBUI_URL})...")
        try:
            # Conexión persistente SSE
            streams = await app.state.exit_stack.enter_async_context(sse_client(MCP_OPENWEBUI_URL))
            session = await app.state.exit_stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            
            # Construir el agente usando la sesión
            mcp_agent_instance = await mcp_agent_module.build_mcp_worker_agent(session)
            logger.info("✅ MCP OpenWebUI Agent listo.")
        except Exception as e:
            logger.error(f"⚠️ Fallo al conectar con p.py: {e}. El orquestador funcionará sin OpenWebUI.")
            mcp_agent_instance = None

        # --- 3. PREPARAR EL ORQUESTADOR ---
        # Guardamos los agentes en el estado de la app para usarlos en cada petición
        app.state.orchestrator = build_orchestrator()
        app.state.agents_config = {
            "brain_agent": brain_agent_instance,
            "mcp_agent": mcp_agent_instance
        }
        logger.info("🤖 ORQUESTADOR OPERATIVO.")

    except Exception as e:
        logger.error(f"❌ Error fatal en inicio: {e}")
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
    # Pasamos 'config' en el parámetro 'configurable' para que los nodos accedan a los agentes
    result = await orchestrator.ainvoke(inputs, config={"configurable": config})
    
    return {
        "response": result["final_response"],
        "routed_to": result.get("next_agent", "unknown")
    }