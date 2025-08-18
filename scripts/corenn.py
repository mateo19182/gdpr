from corenn_py import CoreNN
import numpy as np

db = CoreNN.open("data/db")

print(db)
keys = [
    "my_entry_1",
    "my_entry_2",
]
vectors = np.array([
    [0.3, 0.6, 0.9],
    [0.4, 1.1, 0.0],
], dtype=np.float32)

db.insert_f32(keys, vectors)
print("Inserted vectors.")
    
queries = np.array([
    [1.0, 1.3, 1.7],
    [7.3, 2.5, 0.0],
], dtype=np.float32)

k100 = db.query_f32(queries, 100)
print("Queried vectors.")
print(k100)