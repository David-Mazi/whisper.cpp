from ollama import chat
from ollama import ChatResponse
from ollama import Client
import pandas as pd


# Placeholder for whatever raw text you want to feed the model.
rawText = """
INPUT TEXT
"""

# Pre-instruction: injects the raw input text via an f-string, then the schema
# (postInputText) is appended after it. Keep this generic and swap in your own text.
def preInputText(inputText):
    return f"""
            Can you please convert the following text and format the results following the inputSchema provided below?
            Input Text: {inputText}"""

# Generic example of the "forced JSON response" pattern.
#
# This is a stripped-down, non-domain-specific schema that only demonstrates
# the SHAPE the model expects:
#   - inputSchema -> json -> type: object -> properties
#   - each property has a type, description and (optionally) rules
#   - examples of an enum field, a nested object field, and an array-of-objects field
#   - a top-level required list
#
# Fill in your own property names/types for your use case.
postInputText = """
          The inputSchema is in the inputSchema key in the object below:
          {
            "name": "summarize_record",
            "description": "Summarize record data.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string or null",
                            "description": "A free-text name field",
                            "rules": ["Set this field to null if value is not available"]
                        },
                        "category": {
                            "type": "string or null",
                            "description": "A category represented as a string",
                            "enum": [{1: "TYPE_A"}, {2: "TYPE_B"}, {3: "TYPE_C"}],
                            "rules": ["Set this field to 'TYPE_A' if value is not available or part of provided enum"]
                        },
                        "date": {
                            "type": "object or null",
                            "description": "Example of a nested object field",
                            "properties": {
                                "year": {
                                    "type": "integer",
                                    "description": "Year"
                                },
                                "month": {
                                    "type": "integer",
                                    "description": "Month"
                                },
                                "day": {
                                    "type": "integer",
                                    "description": "Day"
                                }
                            },
                            "rules": ["Set this field to null if value is not available"]
                        },
                        "measurement": {
                            "type": "object or null",
                            "description": "Example of a value + unit object with an enum on the unit",
                            "properties": {
                                "units": {
                                    "type": "string",
                                    "description": "Units represented as a string",
                                    "enum": [{1: "UNIT_A"}, {2: "UNIT_B"}]
                                },
                                "value": {
                                    "type": "integer",
                                    "description": "Numeric value of the measurement"
                                }
                            },
                            "rules": ["Set this field to null if value is not available"]
                        },
                        "identifiers": {
                            "type": "array or null",
                            "description": "Example of an array of objects, each with an id type and a value",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "integer",
                                        "description": "ID Types",
                                        "enum": [{1: "TYPE_ONE"}, {2: "TYPE_TWO"}]
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Value representing the unique id"
                                    }
                                }
                            },
                            "rules": ["Set this field to null if value is not available"]
                        }
                    },
                    "required": [
                        "category"
                    ]
                }
            }
          }
          Repeat rules here if the model is having an issue picking them up, e.g.:
          For any missing information set value to null
          Please return just the json with no other surrounding text
          Please maintain the order of the inputSchema provided when generating the response json
          If a date is in the form xx-xx-xxxx, xx/xx/xxxx, or xx xx xxxx, assume the first number represents the month unless otherwise stated.
          """


client = Client()

def getResponse(inputToModel):
    prompt = preInputText(inputToModel) + postInputText

    response = client.generate(
        prompt=prompt,
        model="deepseek-r1:7b",
        options={"temperature": 0.6}
    )

    return response.response

print(getResponse(rawText))


def read_file_and_count_rows(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, TSV or XLSX file.")

    return len(df)

# file_path = "data.xlsx"  # Replace with your actual file path
# row_count = read_file_and_count_rows(file_path)
# print(f"Number of rows in {file_path}: {row_count}")

# # Read XLSX and run each row through the model
# data = {'Input': [], 'Output': []}

# df = pd.read_excel('data.xlsx')
# for index, row in df.iterrows():
#     textData = ' '.join(row.astype(str))
#     data['Input'].append(textData)
#     data['Output'].append(getResponse(textData))
#     print(f"Row {index+1} Done")

# df = pd.DataFrame(data)

# # Write the DataFrame to an Excel file
# df.to_excel('output.xlsx', index=False)
