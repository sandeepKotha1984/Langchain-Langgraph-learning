from typing import Any

from langchain.tools import tool
from sqlalchemy import text

from app.db.connection import get_db

@tool
def get_leave_balance(employee_code: str) -> dict[str, Any]:
    """
    Get the current leave balance for an employee.
    """

    query = text("""
        SELECT
            e.employee_code,
            e.name,
            lb.annual_leave,
            lb.sick_leave
        FROM employees e
        JOIN leave_balances lb
            ON e.id = lb.employee_id
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

    return {
        "employee_code": result["employee_code"],
        "employee_name": result["name"],
        "annual_leave": result["annual_leave"],
        "sick_leave": result["sick_leave"],
    }