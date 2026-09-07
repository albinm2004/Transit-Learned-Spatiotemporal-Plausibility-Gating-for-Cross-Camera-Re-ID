# Cross-Camera Person Re-ID with Vector Search

A standalone module for the cross-camera identity stage of a multi-camera person tracking system.

## What this does

`YOLO (teammate) -> person crops -> OSNet Re-ID embedding -> FAISS vector database -> similarity search -> global person identity`

YOLO is intentionally NOT included. Your teammate can provide person crops from each camera. The module stores embeddings plus metadata (`camera_id`, `track_id`, `timestamp`, image path) and retrieves the closest known appearances.

## Quick start

### 1. Create environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

If `torchreid` is unavailable from PyPI in your environment:

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### 2. Enroll the target person

Put several good crops of the target person in:

```text
data/gallery/target/
  001.jpg
  002.jpg
  003.jpg
```

Then:

```bash
python -m src.enroll --gallery data/gallery/target
```

This creates `outputs/target_embedding.npy`.

### 3. Index camera observations

Organize YOLO-generated person crops like this:

```text
data/cameras/
  cam01/
    track_12_001.jpg
    track_12_002.jpg
  cam02/
    track_7_001.jpg
    track_7_002.jpg
```

You can also supply metadata using a CSV. See `examples/observations.csv`.

Run:

```bash
python -m src.index --input data/cameras --db outputs/vector_db
```

### 4. Search for the target

```bash
python -m src.search --db outputs/vector_db --query outputs/target_embedding.npy --top-k 10
```

The output ranks the closest observations by cosine similarity.

## Connecting your teammate's YOLO module

The clean interface is simply:

```text
YOLO detection
    -> crop person bounding box
    -> save crop.jpg
    -> call ReIDEmbedder.embed_image(crop.jpg)
    -> VectorDB.add(vector, metadata)
```

If YOLO already has a tracker, pass its values as metadata:

- `camera_id`
- `track_id`
- `frame_id`
- `timestamp`
- `bbox`

The vector database does not need to know anything about YOLO internals.

## Important design choice

The vector DB is a retrieval layer, not the final truth mechanism. A similarity threshold is configurable. In a later version you can add temporal/camera constraints without changing the embedding database.

## Project structure

```text
src/
  embedder.py       OSNet Re-ID embedding
  vector_db.py      FAISS index + JSON metadata
  enroll.py         Build target identity embedding
  index.py          Index camera observations
  search.py         Search for target
  pipeline.py       Python API for integration with YOLO
examples/
  observations.csv
requirements.txt
```
