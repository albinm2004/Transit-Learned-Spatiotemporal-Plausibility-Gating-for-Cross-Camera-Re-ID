# YOLO integration contract

Your teammate's component should output a **person crop** for each YOLO detection/track.

Minimum:

```python
from src.pipeline import CrossCameraReID

reid = CrossCameraReID("outputs/live_vector_db")

# after YOLO detects/tracks a person and saves the crop:
results = reid.identify(
    "path/to/person_crop.jpg",
    top_k=5,
    threshold=0.55,
)

print(results)
```

To add a known observation to the database:

```python
reid.add_observation(
    "path/to/person_crop.jpg",
    {
        "camera_id": "cam03",
        "track_id": 42,
        "frame_id": 1820,
        "timestamp": 60.67,
    },
)
```

## Recommended flow

1. YOLO detects a person.
2. Tracker assigns a local track ID.
3. Save/use the person bounding-box crop.
4. `identify()` searches the vector DB.
5. If the similarity is above the chosen threshold, associate the observation with the returned global identity.
6. Otherwise create/store a new identity candidate.

For your target-person-only use case, enroll the target first and compare every new crop against the target embedding.

## Notes

- Similarity is cosine similarity because embeddings are L2-normalized.
- The default threshold is deliberately configurable; do not claim a universal value without validating it on your camera data.
- Use multiple target gallery images rather than one image when possible.
- The system does not use face recognition; it is appearance-based person Re-ID.
