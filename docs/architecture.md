# Architecture

The system separates the fixed model from identity data:

```mermaid
flowchart LR
  Camera --> Detect[OpenCV YuNet] --> Align[OpenCV SFace alignment] --> Embed[Fixed OpenCV SFace model]
  Embed --> Search[FAISS / exact vector search]
  Search --> Threshold{Configured threshold}
  Threshold --> Known
  Threshold --> Unknown
  IdentityDB[(SQLite)] --> Search
```

Adding a person adds one normalized vector and metadata. Deleting a person removes or deactivates metadata and vector data. Neither operation retrains the embedding model.
