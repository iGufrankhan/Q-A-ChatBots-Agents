import sqlite3

# Connect to SQLite database (creates file if not exists)
connection = sqlite3.connect("student.db")
cursor = connection.cursor()

# Create table safely
cursor.execute("""
CREATE TABLE IF NOT EXISTS STUDENT(
    NAME TEXT,
    CLASS TEXT,
    SECTION TEXT,
    MARKS INTEGER
)
""")

# Optional: Clear old data (so duplicates don’t happen)
cursor.execute("DELETE FROM STUDENT")

# Insert records safely
students = [
    ('Krish','Data Science','A',90),
    ('John','Data Science','B',100),
    ('Mukesh','Data Science','A',86),
    ('Jacob','DEVOPS','A',50),
    ('Dipesh','DEVOPS','A',35)
]

cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", students)

# Display records
print("The inserted records are:")
for row in cursor.execute("SELECT * FROM STUDENT"):
    print(row)

# Commit and close
connection.commit()
connection.close()