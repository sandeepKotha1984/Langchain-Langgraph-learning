from typing import Any

from langchain.tools import tool
from sqlalchemy import text

from app.db.connection import get_db

@tool
def get_attendance(employee_code: str) -> dict:
    """
    Get attendance details for an employee.
    """

    query = text("""
        SELECT
            e.employee_code,
            e.name,
            a.status,
            a.attendance_date
        FROM employees e
        JOIN attendance a
            ON e.id = a.employee_id
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
        "status": result["status"],
        "attendance_date": str(result["attendance_date"])
    }


