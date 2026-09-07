"""Print the basic commands for this project."""
print("1) pip install -r requirements.txt")
print("2) python -m src.enroll --gallery data/gallery/target")
print("3) python -m src.index --input data/cameras --db outputs/vector_db")
print("4) python -m src.search --db outputs/vector_db --query outputs/target_embedding.npy --top-k 10")
