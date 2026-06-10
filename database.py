import sqlite3

conn = sqlite3.connect("brain_tumor.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    prediction TEXT,
    confidence REAL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database Created Successfully")
