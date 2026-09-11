# Real-Time Single-Image Face Recognition System for CCTV and Webcam

> **Academic project:** enroll a person with one reference photo, recognize faces from USB or IP cameras, and add or delete identities without retraining the recognition model.

## Contents

1. [Project overview](#project-overview)
2. [Problem statement](#problem-statement)
3. [Objectives](#objectives)
4. [Core design principle](#core-design-principle)
5. [Features](#features)
6. [Technology stack](#technology-stack)
7. [System architecture](#system-architecture)
8. [Workflows](#workflows)
9. [Database design](#database-design)
10. [Project structure](#project-structure)
11. [Installation and setup](#installation-and-setup)
12. [Running the application](#running-the-application)
13. [Camera configuration](#camera-configuration)
14. [Enrollment and identity management](#enrollment-and-identity-management)
15. [Recognition logic and UNKNOWN handling](#recognition-logic-and-unknown-handling)
16. [API reference](#api-reference)
17. [Evaluation](#evaluation)
18. [Testing](#testing)
19. [Security and privacy](#security-and-privacy)
20. [Performance](#performance)
21. [Current implementation status](#current-implementation-status)
22. [Troubleshooting](#troubleshooting)
23. [Limitations and future enhancements](#limitations-and-future-enhancements)

---

## Project overview

This system performs **face recognition, not face classification**. A fixed, pre-trained InsightFace ArcFace model converts every detected face into a numeric embedding. The application stores that embedding with identity information in SQLite and indexes it in FAISS for fast similarity search.

Adding a new person creates one embedding and adds it to the database/vector index. Deleting a person deactivates database records and removes the matching vector entry. The ArcFace model is never trained, fine-tuned, or restarted merely because identities change.

```text
Camera frame -> Face detection -> Face alignment -> ArcFace embedding
             -> FAISS similarity search -> Threshold decision -> KNOWN / UNKNOWN
```

## Problem statement

Many face-recognition projects train a classifier class for each person. Adding or deleting a person then requires new data and model retraining. That approach is unsuitable for a dynamic CCTV deployment.

This project uses embedding-based recognition instead. The pre-trained model is fixed, while identity data is held in a persistent database and vector index. This permits immediate identity add/delete operations and supports recognition of multiple faces in a camera frame.

## Objectives

- Enroll a person from exactly one valid face image.
- Support USB/webcams and RTSP/IP CCTV cameras.
- Detect and recognize multiple faces independently per frame.
- Return `UNKNOWN` unless a similarity score meets the configured threshold.
- Persist identities, face-image metadata, embeddings, cameras, audit logs, and recognition events.
- Use FAISS inner-product search over normalized embeddings.
- Allow single/batch deletion without model retraining.
- Restore vector-search state from stored embeddings after a backend restart.
- Generate evaluation metrics only from real experimental data.
- Provide a FastAPI backend, OpenAPI documentation, and React dashboard.

## Core design principle

### Model and identity data are separate

| Component | Responsibility | Changes when a person is added/deleted? |
| --- | --- | --- |
| Pre-trained InsightFace ArcFace model | Face image -> normalized embedding | No |
| SQLite database | Identity metadata, image records, embedding relation | Yes |
| FAISS vector index | Vector ID -> facial embedding | Yes |

```text
Add person    = database + vector-index operation
Delete person = database + vector-index operation
Model retraining = never required for either operation
```

## Features

### Identity enrollment

- Full name, unique identity code, optional metadata, and reference photo.
- Exactly one detectable face is required.
- Required validation responses:
  - `No face detected. Please upload another image.`
  - `Multiple faces detected. Please upload an image containing only one person.`
- Decode image, detect face, generate/normalize embedding, store image/metadata, update FAISS, and log enrollment.

### Batch enrollment

- Processes each uploaded image independently.
- Reports total files, successful files, failed files, and a reason for every failure.
- Does not train or retrain the model.

### Live recognition

- USB webcam and RTSP source support.
- Multiple-face detection per frame.
- Bounding boxes, names, similarity values, and `UNKNOWN` labels.
- FPS and latency instrumentation.
- Camera-specific recognition event records.

### Camera management

- Multiple camera configurations.
- USB device index or RTSP URL.
- Enabled/disabled status.
- Independent camera-worker lifecycle.

### Identity deletion

- Delete one identity, selected identities, or reference-image records.
- Deactivate records and remove matching vector IDs.
- Rebuild FAISS safely from active embeddings when needed.
- Index rebuilding is **not** model retraining.

### Evaluation

- Detection rate, accuracy, precision, recall, F1, FAR, FRR, FPS, average latency, median latency, and p95 latency.
- Threshold sweep using actual genuine/impostor similarity scores.
- Confusion-matrix-ready predictions including `UNKNOWN`.

## Technology stack

| Layer | Technology |
| --- | --- |
| Backend API | Python 3.12, FastAPI, Uvicorn |
| Face detection and embedding | OpenCV, InsightFace `buffalo_l`, SCRFD, ArcFace |
| Inference runtime | ONNX Runtime CPU |
| Vector similarity search | FAISS CPU, normalized inner product / cosine similarity |
| Relational data | SQLite, SQLAlchemy ORM |
| Frontend | React, Vite |
| Evaluation | NumPy, pandas, scikit-learn, matplotlib |
| Tests | pytest, FastAPI TestClient |

## System architecture

```mermaid
flowchart TB
    UI[React Dashboard]
    API[FastAPI Backend]
    UI <--> API

    subgraph Input[Camera sources]
        USB[USB / Webcam]
        RTSP[RTSP IP CCTV]
    end

    subgraph Processing[Recognition pipeline]
        Capture[Camera capture worker]
        Detect[SCRFD face detection]
        Align[Face alignment / preprocessing]
        Embed[Fixed ArcFace embedding model]
        Normalize[L2 normalize embedding]
        Search[FAISS inner product search]
        Decision{Similarity >= threshold?}
        Known[KNOWN identity]
        Unknown[UNKNOWN]
    end

    subgraph Storage[Persistent storage]
        DB[(SQLite)]
        Index[(FAISS vector index)]
        Files[Reference image storage]
    end

    USB --> Capture
    RTSP --> Capture
    Capture --> Detect --> Align --> Embed --> Normalize --> Search --> Decision
    Decision -->|Yes| Known
    Decision -->|No| Unknown
    DB <--> Index
    DB <--> Files
    API --> Capture
    API --> DB
    API --> Index
    Known --> DB
    Unknown --> DB
```

### Multi-camera architecture

```mermaid
flowchart LR
    CAM1[CAM-01: USB / RTSP] --> W1[Recognition worker 1]
    CAM2[CAM-02: USB / RTSP] --> W2[Recognition worker 2]
    CAM3[CAM-03: USB / RTSP] --> W3[Recognition worker 3]
    W1 --> Shared[Shared identity database and FAISS index]
    W2 --> Shared
    W3 --> Shared
    Shared --> Events[(Recognition events with camera ID)]
```

## Workflows

### Single-image enrollment flow

```mermaid
flowchart TD
    A[Enter name, ID, metadata and upload photo] --> B[Validate type, size and image decoding]
    B --> C[Detect faces]
    C --> D{Exactly one face?}
    D -->|No faces| E[Return: No face detected]
    D -->|More than one| F[Return: Multiple faces detected]
    D -->|One face| G[Align and generate ArcFace embedding]
    G --> H[Normalize embedding]
    H --> I[Store identity, image record and embedding relation]
    I --> J[Add vector ID to FAISS]
    J --> K[Create ENROLL audit log]
    K --> L[Enrollment successful: no retraining]
```

### Recognition flow

```mermaid
flowchart TD
    A[Read camera frame] --> B[Detect all faces]
    B --> C{Any faces?}
    C -->|No| Z[Send/display frame]
    C -->|Yes| D[For each face: align and embed]
    D --> E[Normalize embedding]
    E --> F[Search nearest active vector in FAISS]
    F --> G{Score >= threshold?}
    G -->|Yes| H[Draw identity and similarity]
    G -->|No| I[Draw UNKNOWN and similarity]
    H --> J[Record KNOWN event]
    I --> K[Record UNKNOWN event]
    J --> Z
    K --> Z
```

### Add/delete without retraining sequence

```mermaid
sequenceDiagram
    participant User
    participant Service as FastAPI / service
    participant Model as Fixed ArcFace model
    participant DB as SQLite
    participant Index as FAISS

    User->>Service: Enroll person A with one photo
    Service->>Model: Detect and embed
    Model-->>Service: Normalized embedding A
    Service->>DB: Save identity/image/embedding metadata
    Service->>Index: Add vector A
    Note over Model: No training or retraining

    User->>Service: Enroll person B while system is running
    Service->>Model: Detect and embed
    Model-->>Service: Normalized embedding B
    Service->>DB: Save person B
    Service->>Index: Add vector B

    User->>Service: Delete person B
    Service->>DB: Deactivate person B and image records
    Service->>Index: Remove vector B or rebuild active index
    Note over Model: No training or retraining
```

## Database design

```mermaid
flowchart LR
    I[(identities<br/>id, identity_code, name<br/>metadata, status, timestamps)]
    F[(face_images<br/>id, identity_id, image_path<br/>embedding_id, embedding, status)]
    C[(cameras<br/>id, camera_name, camera_type<br/>device_index, rtsp_url, status)]
    R[(recognition_events<br/>id, identity_id, camera_id<br/>similarity_score, result, timestamp)]
    A[(audit_logs<br/>id, identity_id, operation<br/>description, timestamp)]

    I -->|one identity owns many images| F
    I -->|identity may be recognized| R
    C -->|camera produces events| R
    I -->|operations may affect identity| A
```

## Project structure

```text
assignment_image/
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI routes
│   │   ├── database/            # SQLAlchemy models and repositories
│   │   ├── models/              # Detector and ArcFace embedder
│   │   ├── services/            # Enrollment, recognition, camera, vector, evaluation services
│   │   └── main.py              # Backend entry point
│   ├── tests/                   # pytest unit/integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/                     # React dashboard shell and styles
│   ├── package.json
│   └── vite.config.js
├── docs/
│   ├── api.md
│   ├── architecture.md
│   └── evaluation.md
├── .env.example
└── README.md
```

## Installation and setup

### Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer
- USB camera or valid RTSP source for live testing

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

InsightFace downloads `buffalo_l` on its first model initialization. This workspace has verified weights at:

```text
C:\Users\padma\.insightface\models\buffalo_l
```

### Frontend

```powershell
cd frontend
npm install
```

The project pins Vite `6.3.5` and React `18.3.1` for compatibility with Node.js `20.15.0`.

### Configuration

Copy `.env.example` to `.env` and adjust as required:

```dotenv
DATABASE_URL=sqlite:///./data/face_recognition.db
DATA_DIR=./data
MODEL_NAME=buffalo_l
RECOGNITION_THRESHOLD=0.60
FRAME_SKIP=2
MAX_UPLOAD_MB=10
```

The threshold is configurable. `0.60` is only a starting value, not a universal recommendation. Select the final value from a validation experiment.

## Running the application

### Backend

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Interactive OpenAPI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health check: `http://127.0.0.1:8000/health`

### Frontend

```powershell
cd frontend
npm run dev
```

Create a production build with:

```powershell
npm run build
```

## Camera configuration

### USB/webcam

```text
Camera name: Main Webcam
Camera type: USB
Device index: 0
```

### RTSP/IP CCTV camera

```text
Camera name: Entrance CCTV
Camera type: RTSP
RTSP URL: rtsp://username:password@camera-ip/stream
```

The browser must not directly connect to RTSP. The intended deployment is camera -> backend processing -> browser-compatible MJPEG, WebSocket, or WebRTC. Do not expose RTSP credentials in ordinary UI views or logs.

## Enrollment and identity management

Required enrollment data:

- Full name
- Unique identity/student/employee code
- Optional metadata
- One reference image containing one face

Example batch report:

```text
Total files: 100
Successful: 94
Failed: 6

Failures:
image23.jpg - No face detected
image41.jpg - Multiple faces detected
image52.jpg - Invalid image
```

Deleting an image or identity must make its vector unavailable for future recognition. If a safe FAISS rebuild is used, it rebuilds only the vector index from active database embeddings. It does not train the face model.

## Recognition logic and UNKNOWN handling

Embeddings are L2-normalized, so FAISS inner-product score approximates cosine similarity:

```text
cosine_similarity(a, b) ~= normalized_a dot normalized_b
```

For each detected face:

1. Generate an embedding.
2. Search the closest active vector.
3. Compare the score against the configured threshold.
4. Return the identity only when the threshold passes.
5. Otherwise return `UNKNOWN`.

```text
Closest identity: Uday
Similarity: 0.48
Threshold: 0.70
Result: UNKNOWN
```

The closest database result is never accepted automatically.

## API reference

FastAPI publishes the live contract at `/docs`. Current routes:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Application health response |
| `GET` | `/api/identities` | List active identities |
| `POST` | `/api/identities` | Create identity metadata |
| `DELETE` | `/api/identities/{id}` | Soft-delete one identity |
| `GET` | `/api/cameras` | List cameras |
| `POST` | `/api/cameras` | Create camera configuration |
| `GET` | `/api/settings` | Read recognition settings |
| `PUT` | `/api/settings` | Update safe settings |
| `GET` | `/api/statistics` | Identity, camera, event counts |

The original project specification also requires multipart image enrollment, batch enrollment, deletion of a single image, recognition history, browser streaming, and evaluation endpoints. Their integration status is documented below.

## Evaluation

### Dataset layout

Do not use an enrollment image as the primary recognition test image.

```text
dataset/
├── enrollment/
│   ├── uday.jpg
│   ├── arun.jpg
│   └── rahul.jpg
├── test/
│   ├── known/
│   │   ├── uday/test1.jpg
│   │   ├── arun/test1.jpg
│   │   └── rahul/test1.jpg
│   └── unknown/
│       ├── unknown1.jpg
│       └── unknown2.jpg
└── conditions/
    ├── normal/
    ├── low_light/
    ├── side_pose/
    ├── low_resolution/
    ├── distance/
    └── occlusion/
```

### Metrics

| Metric | Formula |
| --- | --- |
| Detection rate | detected faces / actual faces |
| Recognition accuracy | correct predictions / total predictions |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 x precision x recall / (precision + recall) |
| FAR | false acceptances / impostor attempts |
| FRR | false rejections / genuine attempts |
| FPS | processed frames / elapsed time |

Report average, median, and p95 latency from measured frame-processing times.

### Threshold analysis

Evaluate genuine and impostor samples across measured thresholds, for example:

```text
0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
```

Compute FAR, FRR, precision, recall, and F1 at each threshold. Recommend a threshold only from actual validation results.

## Testing

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

Verified result in this workspace:

```text
31 passed
```

Test coverage includes detection handling, embedding normalization, enrollment validation, SQLite lifecycle, FAISS persistence, thresholded `UNKNOWN`, USB/RTSP errors, batch reports, deletion/vector removal, events, multi-camera workers, metrics, threshold analysis, API behavior, the no-retraining acceptance workflow, and performance instrumentation.

## Security and privacy

Face images and embeddings are sensitive biometric data. Before real deployment, implement or verify:

- Consent and lawful data processing.
- Authentication and role-based access.
- Encryption for reference images, database backups, and RTSP credentials.
- HTTPS/TLS for network traffic.
- File MIME/type, size, image-content validation, and sanitized filenames.
- No uploaded-file execution.
- No passwords or secrets in normal logs, API responses, or UI.
- Retention/deletion policy and audit records.

## Performance

The project includes FPS/latency instrumentation. It does not claim unmeasured speed or accuracy.

After testing real hardware/cameras, report:

```text
Hardware
CPU: <measured machine>
GPU: <if used>
RAM: <measured machine>

Model
Detector: SCRFD
Embedding model: ArcFace / buffalo_l
Resolution: <camera resolution>
Registered identities: <count>

Measured
Detection FPS: <measured>
Recognition FPS: <measured>
Average latency: <measured>
Median latency: <measured>
95th percentile latency: <measured>
```

Potential optimizations include frame skipping, reduced detection resolution, batch embeddings, face tracking, asynchronous capture, and one worker per camera. Correct `UNKNOWN` behavior must remain unchanged.

## Current implementation status

This table separates the full assignment specification from the code currently wired end-to-end in this workspace.

| Area | Status | Notes |
| --- | --- | --- |
| InsightFace `buffalo_l` model | Verified | SCRFD and ArcFace weights load on CPU. |
| Face detection / ArcFace embedding | Implemented | Fixed model; embeddings normalized. |
| SQLite schemas/repositories | Implemented | Identity, image, camera, event, audit schemas exist. |
| FAISS vector store | Verified | FAISS active; persistence/rebuild tested. |
| Similarity threshold / `UNKNOWN` | Implemented | Low score returns `UNKNOWN`. |
| Webcam/RTSP capture services | Implemented | Requires real camera for live validation. |
| Multi-camera worker manager | Implemented | Independent start/stop registry. |
| Batch validation | Implemented | Per-file validation report. |
| Deletion/vector removal | Implemented | Soft-delete and vector removal tested. |
| Evaluation helpers | Implemented | Requires real experimental samples. |
| API metadata/cameras/settings | Implemented | Current routes listed above. |
| Multipart photo enrollment API | Pending integration | Service exists, full upload route is not wired. |
| Batch enrollment API | Pending integration | Validation exists, upload/persistence route remains. |
| MJPEG/WebSocket browser stream | Pending integration | Capture/processing exist; stream route remains. |
| Full React CRUD/live/evaluation views | Partial | Dashboard shell exists; functional views need API integration. |
| Real camera/dataset benchmark | Not measured | No fabricated accuracy/FAR/FRR/FPS/latency values. |
| Authentication/encryption | Pending hardening | Required before deployment. |

## Troubleshooting

| Problem | Likely cause | Resolution |
| --- | --- | --- |
| `InsightFace is not installed` | Missing dependencies | Activate `.venv` and install `backend/requirements.txt`. |
| Model fails to load | Interrupted download/missing model files | Re-run model initialization and verify the `buffalo_l` ONNX files. |
| No face detected | Poor image/light or corrupt image | Upload a clear image containing one visible face. |
| Multiple-face error | Reference image includes multiple people | Use an image containing only the enrolled person. |
| Camera unavailable | Wrong USB index, busy device, bad RTSP/network | Verify camera details and test source independently. |
| RTSP rejected | Invalid URL | Use `rtsp://host/path` with valid credentials/path. |
| Frontend build fails | Dependency mismatch | Run `npm install`; project uses Vite 6 for Node 20.15 compatibility. |
| Result is UNKNOWN | Score below threshold | Check image quality and determine threshold from validation data. |

## Limitations and future enhancements

Current limitations:

- One image gives less pose/lighting coverage than multi-image enrollment.
- CCTV quality, distance, occlusion, lighting, compression, and pose can reduce recognition quality.
- CPU-only inference can limit camera count and frame rate.
- Biometric processing has legal, ethical, privacy, and bias considerations.
- Real deployment requires authentication, encrypted secret management, HTTPS, and retention controls.

Future enhancements:

- Full multipart enrollment, batch, deletion, history, evaluation, and stream APIs.
- React forms, live video, identity table, filters, settings, and charts.
- MJPEG/WebSocket/WebRTC streaming.
- Face tracking and recognition-every-N-frames optimization.
- GPU inference.
- PostgreSQL, Redis/message queues, and distributed camera processing.
- Role-based access, alerts, attendance integration, access control, Docker, and migrations.

## Conclusion

The project is built around the required separation between a fixed face-embedding model and a dynamic identity/vector database. This allows one photo -> embedding -> searchable identity, live camera frame -> detection -> embedding -> thresholded match -> known or `UNKNOWN`, and immediate add/delete operations without model retraining.

For the final academic demonstration, complete the pending API/UI wiring, collect a separate validation dataset, test the real target cameras, and report only experimentally measured performance and recognition results.
