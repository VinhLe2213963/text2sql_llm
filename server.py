from utils import *
from retriever_model import Retriever
import os 
from google import genai
import torch
from llm_model import Gemini
from openai import OpenAI


test_dataset = pd.read_pickle('test_dataset.pkl')

small_dataset = pd.read_pickle("small_dataset.pkl")


table_path = "spider_data/test_tables.json"

df = pd.read_csv("vnese_examples_database.csv")
all_questions = df['question'].dropna().tolist()

all_vnese_questions = pd.read_csv('all_vnese_questions.csv')['vnese_question'].tolist()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
retriever = Retriever(all_vnese_questions, torch.load('vnese_question_embeddings.pt', map_location='cpu'), None, "gemini/embedding-001", client)

def get_sql(question, question_embedding, db_id, db_path, examples_database_path, pre_llm, fin_llm, temperature):
    table_schemas, foreign_key_pairs = get_table_schema(db_id, table_path)
    table_schemas_prompt = generate_vn_table_schema_prompt(table_schemas)
    foreign_key_prompt = generate_vn_foreign_key_prompt(foreign_key_pairs)
    cv_ref = generate_vn_cell_value_reference(db_id, table_schemas, db_path)

    question_examples = retriever.get_examples(question_embedding)

    df = pd.read_csv(examples_database_path)

    fewshot_examples = {
        'questions': question_examples,
        'queries': []
    }

    for example in question_examples:
        match = df.loc[df['vnese_question'] == example, 'query']
        fewshot_examples['queries'].append(match.iloc[0])

    formatted_fewshot_examples = format_fewshot_examples(fewshot_examples)

    # prompt = generate_vn_prompt(question, formatted_fewshot_examples, table_schemas_prompt, foreign_key_prompt, cv_ref)
    # print("-------------------------------------------------------")
    # print(prompt)
    # print("-------------------------------------------------------")

    # pre_sql = ""
    # while True:
    #     try:
    #         pre_sql = pre_llm.generate(prompt, 1)
    #         break
    #     except Exception as e:
    #         print(e)
    #         time.sleep(30)

    # print("-------------------------------------------------------")
    # print(pre_sql)
    # print("-------------------------------------------------------")

    # keep_tables = Parser(pre_sql).tables

    # table_schemas = {
    #     table: cols
    #     for table, cols in table_schemas.items()
    #     if table in keep_tables
    # }

    # filtered_fk = filter_foreign_keys(table_schemas, foreign_key_pairs)

    # table_schemas_prompt = generate_vn_table_schema_prompt(table_schemas)
    # foreign_key_prompt = generate_vn_foreign_key_prompt(filtered_fk)
    # cv_ref = generate_vn_cell_value_reference(db_id, table_schemas, db_path)

    # final_prompt = generate_prompt(question, formatted_fewshot_examples, table_schemas_prompt, foreign_key_prompt, cv_ref)

    fin_sql = []

    # while True:
    #     try:
    #         fin_sql = fin_llm.generate(final_prompt, temperature)
    #         break
    #     except Exception as e:
    #         print(e)
    #         time.sleep(30)

    for llm in fin_llm:
      final_prompt = generate_vn_prompt(question, formatted_fewshot_examples, table_schemas_prompt, foreign_key_prompt, cv_ref)
      print("-------------------------------------------------------")
      print(final_prompt)
      print("-------------------------------------------------------")
      while True:
        try:
          sql = llm.generate(final_prompt, temperature)

          sql_error_list = []
          for attemp in range(5):
              try:
                  conn = sqlite3.connect(db_path)
                  cursor = conn.cursor()
                  res = cursor.execute(sql)
                  res = cursor.fetchall()
                  break
              except Exception as e:
                  sql_error_list.append((sql, e))
                  error_prompt = generate_vn_error(sql_error_list)
                  final_prompt = generate_prompt(question, formatted_fewshot_examples, table_schemas_prompt, foreign_key_prompt, cv_ref, error_prompt)
                  print(final_prompt)
                  try:
                      sql = llm.generate(final_prompt, temperature)
                  except Exception as e:
                      print(e)
                      time.sleep(30)
                  continue

          fin_sql.append(sql)
          break
        except Exception as e:
          print(e)
          time.sleep(30)

    return fin_sql

gemini25 = Gemini("gemini-2.5-flash-preview-05-20", os.getenv("GEMINI_API_KEY"))

# exec_acc = 0

# from tqdm import tqdm

# start, end = 0, 1

# for i in tqdm(range(start,end)):
#     db_id, query, question, difficulty, question_embedding, vnese_question = small_dataset.iloc[i]
#     db_path = f"spider_data/test_database/{db_id}/{db_id}.sqlite"
#     examples_database_path = "vnese_examples_database.csv"
#     gemini_sql = get_sql(vnese_question, retriever.get_embedding(vnese_question), db_id, db_path, examples_database_path, gemini25, [gemini25], 1)
#     try:
#         if compute_execution_accuracy(gemini_sql, query, db_path):
#             exec_acc += 1
#     except Exception as e:
#         print(e)
#     # if compute_execution_accuracy(gemini_sql, query, db_path):
#     #     exec_acc += 1

# print(exec_acc)
# print(exec_acc/end)

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS if your client is running on a different port

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Extract form data
        message = request.form.get('message')
        db_id = request.form.get('dbId')
        context_json = request.form.get('context')

        # Parse context if present
        context = []
        if context_json:
            context = json.loads(context_json)

        # Process uploaded files
        attachments = []
        for key in request.files:
            file = request.files[key]
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                attachments.append({
                    "filename": filename,
                    "filepath": filepath
                })

        print("Received message:", message)
        print("Database ID:", db_id if db_id else "art_1")
        print("Context:", context)
        print("Attachments:", [f["filename"] for f in attachments])

        db_path = f"spider_data/test_database/{db_id}/{db_id}.sqlite"
        vnese_question = message
        examples_database_path = "vnese_examples_database.csv"
        gemini_sql = get_sql(vnese_question, retriever.get_embedding(vnese_question), db_id, db_path, examples_database_path, gemini25, [gemini25], 1)
        
        prompt = f"""
### Query: {gemini_sql}

### Cho một câu truy vấn SQLite như trên, nếu có thể, hãy tách câu truy vấn trên thành những phần nhỏ hơn và ghép lại như cũ để hiểu dễ dàng hơn. Nếu không thể tách, hãy giữ nguyên câu truy vấn trên.

### Hãy trả về một list có dạng như dưới đây trong python:
```python
[
    (Giải thích, câu truy vấn 1),
    (Giải thích, câu truy vấn 2),
    ...,
    (Giải thích, câu truy vấn n),
    (Cuối cùng, câu truy vấn gốc)
]
```
"""
        token = os.getenv("GITHUB_KEY")
        endpoint = "https://models.github.ai/inference"
        model_name = "openai/gpt-4.1-mini"

        client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
        
        res = eval(process_response(response.choices[0].message.content))

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List of queries (label + SQL string)
        queries = res

        final = ""

        # Execute and print each query
        for label, query in queries:
            final += f"\n🔹 {label}"
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            final += tabulate(rows, headers=columns, tablefmt="grid") if rows else "No results."

        conn.close()

        return jsonify({'response': final})

    except Exception as e:
        print("Error handling chat request:", str(e))
        return jsonify({"response": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)