# API

FastAPI generates the authoritative OpenAPI schema at `/docs` and `/openapi.json`.

Current endpoints include:

- `GET/POST /api/identities`
- `DELETE /api/identities/{id}`
- `POST /api/enroll/image` - multipart enrollment with exactly one face
- `POST /api/recognize/image` - recognize an uploaded image against active vectors
- `POST /api/recognize/video` - recognize sampled video frames against active vectors
- `GET/POST /api/cameras`
- `GET/PUT /api/settings`
- `GET /api/statistics`
- `GET /health`

The enrollment route validates the upload, requires exactly one face, stores the normalized embedding, updates FAISS, and creates an audit record. Recognition routes never create identities or embeddings. They search only active vectors already stored by enrollment, apply the configured threshold, and return `UNKNOWN` for low scores. Uploaded videos are temporary processing inputs and are not enrolled automatically.

Live MJPEG/WebSocket streaming, history filters, batch-upload persistence, and evaluation-run endpoints remain future integration work. All upload handlers must retain MIME/size/face-count validation and must not expose raw filesystem paths or RTSP passwords.
