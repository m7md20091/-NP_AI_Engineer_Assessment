import json
import re
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "Employee_ID",
    "Full_Name",
    "Department",
    "Job_Title",
    "Hire_Date",
    "Location",
    "Performance_Rating",
    "Experience_Years",
    "Status",
    "Work_Mode",
    "Salary_INR",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return re.sub(r"\s+", " ", str(value)).strip()


def load_and_clean_employees(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_excel(path, engine="openpyxl")
    missing = EXPECTED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.drop_duplicates(subset=["Employee_ID"]).copy()
    text_columns = [
        "Employee_ID", "Full_Name", "Department", "Job_Title", "Location",
        "Status", "Work_Mode",
    ]
    for column in text_columns:
        frame[column] = frame[column].map(_clean_text)

    for column in ["Performance_Rating", "Experience_Years", "Salary_INR"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if pd.api.types.is_numeric_dtype(frame["Hire_Date"]):
        frame["Hire_Date"] = pd.to_datetime(
            frame["Hire_Date"], unit="D", origin="1899-12-30", errors="coerce"
        )
    else:
        frame["Hire_Date"] = pd.to_datetime(frame["Hire_Date"], errors="coerce")
    frame["Hire_Date"] = frame["Hire_Date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    frame = frame.dropna(subset=["Performance_Rating", "Experience_Years", "Salary_INR"])
    return frame


def employee_to_document(row: pd.Series) -> dict[str, object]:
    salary = int(row["Salary_INR"])
    rating = int(row["Performance_Rating"])
    experience = int(row["Experience_Years"])
    text = (
        f"Employee {row['Full_Name']} ({row['Employee_ID']}) works in the "
        f"{row['Department']} department as {row['Job_Title']}. "
        f"They are located in {row['Location']} and work {row['Work_Mode']}. "
        f"Their employment status is {row['Status']}. They were hired on "
        f"{row['Hire_Date']}, have {experience} years of experience, a performance "
        f"rating of {rating} out of 5, and a salary of INR {salary:,}."
    )
    return {
        "id": row["Employee_ID"],
        "text": text,
        "metadata": {
            "employee_id": row["Employee_ID"],
            "full_name": row["Full_Name"],
            "department": row["Department"],
            "job_title": row["Job_Title"],
            "status": row["Status"],
        },
    }


def ingest_dataset(source: Path, destination: Path) -> list[dict[str, object]]:
    frame = load_and_clean_employees(source)
    documents = [employee_to_document(row) for _, row in frame.iterrows()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    return documents


def load_documents(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
