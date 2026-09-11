# API

FastAPI generates the authoritative OpenAPI schema at `/docs` and `/openapi.json`.

Current endpoints include:

- `GET/POST /api/identities`
- `DELETE /api/identities/{id}`
- `GET/POST /api/cameras`
- `GET/PUT /api/settings`
- `GET /api/statistics`
- `GET /health`

Enrollment, live MJPEG/WebSocket streaming, history filters, and evaluation-run endpoints should be added as their UI modules are connected to the service layer. All upload handlers must retain MIME/size/face-count validation and must not expose raw filesystem paths or RTSP passwords.

