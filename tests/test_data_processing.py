import pandas as pd

from app.data_processing import employee_to_document


def test_employee_to_document_contains_grounding_fields():
    row = pd.Series(
        {
            "Employee_ID": "EMP1", "Full_Name": "Jane Doe", "Department": "IT",
            "Job_Title": "Engineer", "Hire_Date": "2024-01-01", "Location": "Riyadh",
            "Performance_Rating": 5, "Experience_Years": 4, "Status": "Active",
            "Work_Mode": "Hybrid", "Salary_INR": 100000,
        }
    )
    document = employee_to_document(row)
    assert document["id"] == "EMP1"
    assert "Jane Doe" in document["text"]
    assert "IT department" in document["text"]
    assert "INR 100,000" in document["text"]

