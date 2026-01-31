
import sqlite3
import sqlite_vec
import struct

db = sqlite3.connect("database.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False) # Best practice: disable after loading

# 2. Create a Virtual Table for Vectors
# 'vec0' is the module name. We define a column with its dimension (e.g., 4).
# sqlite-vec virtual tables automatically have a 'rowid' that acts as the ID
# You can add additional metadata columns if needed
# db.execute("CREATE VIRTUAL TABLE people_vectors USING vec0(embedding float[512])")

# Example: Insert with auto-generated rowid
import numpy as np

# Sample embedding (512-dimensional for ArcFace)
embedding = np.random.randn(512).astype(np.float32)
embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

# Insert without specifying rowid - SQLite auto-generates it
cursor = db.execute("INSERT INTO people_vectors(embedding) VALUES (?)", (embedding_bytes,))
new_person_id = cursor.lastrowid  # Get the auto-generated rowid
print(f"Inserted new person with ID: {new_person_id}")

# Verify insertion
result = db.execute("SELECT rowid FROM people_vectors WHERE rowid = ?", (new_person_id,)).fetchone()
print(f"Retrieved person ID: {result[0]}")

db.commit()