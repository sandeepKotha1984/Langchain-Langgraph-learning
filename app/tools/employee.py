from typing import Any

from langchain.tools import tool
from sqlalchemy import text

from app.db.connection import get_db

@tool
def get_employee_details(employee_code: str) -> dict:
    """
    Get details for an employee.
    """

    query = text("""
        SELECT
            e.employee_code,
            e.name,
            e.department_id,
            e.email
        FROM employees e
        WHERE e.employee_code = :employee_code
    """)

    with get_db() as db:
        result = db.execute(
            query,
            {"employee_code": employee_code}
        ).mappings().first()

    if result is None:
        return {
            "employee_code": employee_code,
            "error": "Employee not found"
        }
    print(result)
    return {
        "employee_code": result["employee_code"],
        "employee_name": result["name"],
        "employee_email": result["email"],
        "employee_department_id" : result["department_id"]
    }

