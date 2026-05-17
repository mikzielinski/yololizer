# Postman — YOLOlizer API

## Import

1. Open Postman → **Import**
2. Add `YOLOlizer.postman_collection.json`
3. Add `YOLOlizer.local.postman_environment.json` and select **YOLOlizer Local**

## Base URL

| Environment | `baseUrl` |
|-------------|-----------|
| Docker | `http://127.0.0.1:8001` |
| `python run.py` | `http://127.0.0.1:8000` |

## Jobs

After `POST` endpoints that return `job_id`, copy it to the environment variable `job_id` and call the matching **status** request.

## Swagger

Interactive docs: `{{baseUrl}}/docs`
