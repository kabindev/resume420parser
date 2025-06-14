import pandas as pd

data = {
    'filename': ['resume1.pdf'],
    'name': ['John Doe'],
    'email': ['john.doe@example.com'],
    'phone': ['1234567890'],
    'skills': ['Python, SQL'],
    'experience_years': ['5'],
    'education': ["Bachelor's Degree"]
}
df = pd.DataFrame(data)

try:
    df.to_excel("valid.xlsx", index=False, sheet_name="Sheet1")
    print("valid.xlsx created successfully.")
except Exception as e:
    print(f"Error creating valid.xlsx: {e}")
