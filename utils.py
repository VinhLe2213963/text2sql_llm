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
            col_display = f"{col_name}" 
        table_columns[table_idx].append(col_display)

    table_schemas = {
        table_names[i]: cols
        for i, cols in table_columns.items()
    }

    foreign_key_pairs = []
    for from_idx, to_idx in foreign_keys:
        from_table_idx, from_col_name = column_names[from_idx]
        to_table_idx, to_col_name = column_names[to_idx]
        from_table = table_names[from_table_idx]
        to_table = table_names[to_table_idx]
        foreign_key_pairs.append(
            (f"{from_table}.{from_col_name}", f"{to_table}.{to_col_name}")
        )
    
    return table_schemas, foreign_key_pairs

def filter_foreign_keys(table_schemas, foreign_key_pairs):
    valid_tables = set(table_schemas.keys())

    filtered_fks = []
    for from_col, to_col in foreign_key_pairs:
        from_table = from_col.split('.')[0]
        to_table = to_col.split('.')[0]
        if from_table in valid_tables and to_table in valid_tables:
            filtered_fks.append((from_col, to_col))

    return filtered_fks

def generate_table_schema_prompt(table_schemas):
    lines = ["### Sqlite SQL tables , with their properties:"]
    for table in sorted(table_schemas.keys()):
        columns = " , ".join(table_schemas[table])
        lines.append(f"{table} ( {columns} );")
    return "\n".join(lines)

def generate_foreign_key_prompt(foreign_key_pairs):
    lines = ["### Foreign key information of SQLite tables, used for table joins:"]
    for from_col, to_col in foreign_key_pairs:
        from_table, from_column = from_col.split(".")
        to_table, to_column = to_col.split(".")
        line = f"{from_table} ( {from_column} ) REFERENCES {to_table} ( {to_column} );"
        lines.append(line)
    return "\n".join(lines)


def generate_cell_value_reference(db_id, table_schemas, db_path, k=3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    lines = [f"### Here are some data information about database references."]

    for table, columns in sorted(table_schemas.items()):
        values_line = []
        for col in columns:
            try:
                cursor.execute(f"SELECT DISTINCT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL LIMIT 100;")
                results = [row[0] for row in cursor.fetchall()]
                sampled = random.sample(results, min(k, len(results)))
                # Convert all values to string and format them nicely
                sampled_strs = [str(val) for val in sampled]
                values_line.append(f"{col} [{', '.join(sampled_strs)}]")
            except Exception as e:
                # In case of any SQL errors, skip the column
                values_line.append(f"{col} []")

        lines.append(f"{table} ( " + " , ".join(values_line) + " );")

    conn.close()
    return "\n".join(lines)


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