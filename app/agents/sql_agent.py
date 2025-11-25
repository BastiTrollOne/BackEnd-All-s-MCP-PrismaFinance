import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.core.config import settings

sql_logger = logging.getLogger("SQLAgent")

def create_sql_agent_chain(tools: list):
    """
    Creates a LangChain chain that first gets the database schema
    and then uses it to build a valid SQL query.
    """
    sql_logger.info("Creating SQL Agent Chain...")

    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-required",
        model=settings.LLM_MODEL_NAME,
        temperature=0,
        top_p=0.1,
    )

    # Find the necessary tool from the provided list
    get_schema_tool = next((t for t in tools if t.name == "get_database_schema"), None)

    if not get_schema_tool:
        raise ValueError("'get_database_schema' tool not found. Cannot create SQL agent.")

    # Define the prompt template
    template = """
Eres un experto en SQL. Tu tarea es convertir la pregunta de un usuario en una consulta SQL SELECT válida y de solo lectura.

**Instrucciones:**
1.  Analiza la pregunta del usuario y el esquema de la base de datos para construir la consulta más precisa posible.
3.  Utiliza funciones de agregación como `COUNT`, `SUM`, `AVG` y cláusulas como `GROUP BY`, `ORDER BY` cuando la pregunta lo requiera (ej. "top 5", "total de", "promedio de").
3.  Responde únicamente con el código SQL. No incluyas explicaciones, solo el SQL.
4.  Si la pregunta parece ser una consulta SQL con errores de sintaxis (ej. espacios en nombres de columnas), corrígela para que sea válida. Por ejemplo, `Product Id` debe ser `PRODUCT_ID`.

Esquema:
{schema}

Pregunta: {question}

Consulta SQL:
"""
    prompt = ChatPromptTemplate.from_template(template)

    async def _get_schema(input_dict: dict) -> str:
        """Helper async function to invoke the schema tool."""
        return await get_schema_tool.ainvoke({"database_id": input_dict["database_id"]})

    sql_chain = (
        RunnablePassthrough.assign(schema=RunnableLambda(_get_schema))
        | prompt
        | llm
        | StrOutputParser()
    )

    sql_logger.info("SQL Agent Chain created successfully.")
    return sql_chain