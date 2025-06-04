import json
import sqlite3
import pandas as pd
import re
from collections import Counter
import random

def get_table_schema(db_id, table_path):
    with open(table_path, "r") as f:
        tables = json.load(f)

    found_schema = None
    for schema in tables:
        if schema['db_id'] == db_id:
            found_schema = schema
            break

    if not found_schema:
        raise ValueError(f"Schema not found for the given db_id: {db_id} in table_path: {table_path}")

    table_names = found_schema["table_names_original"]
    column_names = found_schema["column_names_original"]
    foreign_keys = found_schema["foreign_keys"]
    primary_keys = found_schema["primary_keys"]

    # Initialize a mapping: table_index -> list of column names
    table_columns = {i: [] for i in range(len(table_names))}

    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue  # skip "*"
        col_display = col_name
        if idx in primary_keys:
            col_display = f"{col_name}"  # bold primary keys if needed
        table_columns[table_idx].append(col_display)

    table_schema = "### Sqlite SQL tables , with their properties:\n"
    for i, table_name in enumerate(table_names):
        cols = " , ".join(table_columns[i])
        table_schema += f"{table_name} ( {cols} );\n"

    foreign_key = "### Foreign key information of SQLite tables, used for table joins:\n"
    for src_idx, tgt_idx in foreign_keys:
        src_table_idx, src_col = column_names[src_idx]
        tgt_table_idx, tgt_col = column_names[tgt_idx]
        foreign_key += f"{table_names[src_table_idx]} ( {src_col} ) REFERENCES {table_names[tgt_table_idx]} ( {tgt_col} );\n"

    return table_schema, foreign_key


def get_cell_value_reference(db_id, table_path, db_path, top_k=3):
    with open(table_path, "r") as f:
        tables = json.load(f)

    found_schema = None
    for schema in tables:
        if schema['db_id'] == db_id:
            found_schema = schema
            break

    if not found_schema:
        raise ValueError(f"Schema not found for the given db_id: {db_id} in table_path: {table_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cell_value_reference = "### Here are some data information about database references.\n"

    for table in found_schema["table_names_original"]:
        try:
            # Get column names
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = [row[1] for row in cursor.fetchall()]

            # Collect distinct values per column
            col_entries = []
            for col in columns:
                try:
                    cursor.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {top_k}')
                    values = [str(row[0]) for row in cursor.fetchall()]
                    formatted_values = ', '.join(values)
                    col_entries.append(f"{col} [{formatted_values}]")
                except Exception as e:
                    col_entries.append(f"{col} [error]")
                    print(f"Error retrieving column '{col}' from table '{table}': {e}")

            # Format output
            output = f"{table} ( " + ' , '.join(col_entries) + " );\n"
            cell_value_reference += output

        except Exception as e:
            print(f"Error reading table '{table}': {e}")

    conn.close()
    return cell_value_reference


def format_fewshot_examples(fewshot_examples):
    lines = []

    for question, query in zip(fewshot_examples['questions'], fewshot_examples['queries']):
        lines.append(f"# {question}")
        lines.append(f"{query}")
        lines.append("")  # empty line for spacing

    return '\n'.join(lines)

def extract_sql_from_llm(llm_output: str) -> str:
    """
    Extract SQL code from a triple-quoted block like ```sql ... ```
    """
    match = re.search(r"```sql\s*(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No valid SQL block found.")
    return match.group(1).strip()


def compute_execution_accuracy(llm_sql, gold_sql, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    llm_res = cursor.execute(llm_sql)
    llm_res = cursor.fetchall()

    gold_res = cursor.execute(gold_sql)
    gold_res = cursor.fetchall()

    return llm_res == gold_res


def majority_vote(rankings):
    hashed_rankings = [tuple(r) for r in rankings]
    
    count = Counter(hashed_rankings)
    max_freq = max(count.values())
    
    tied = [list(r) for r, freq in count.items() if freq == max_freq]
    
    return random.choice(tied)