import re

def normalize_sql(sql: str) -> str:
    """Limpia espacios extra y puntos y coma finales en SQL."""
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";"))

def enforce_limit(sql: str, default_limit: int = 200) -> str:
    """Asegura que la consulta tenga un límite para no saturar."""
    if not sql:
        return sql
    if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {default_limit}"