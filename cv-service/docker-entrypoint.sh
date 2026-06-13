#!/bin/sh
set -e

role="${CONTAINER_ROLE:-api}"

if [ "$#" -gt 0 ]; then
    case "$1" in
        api|worker)
            role="$1"
            shift
            ;;
        celery|uvicorn|python|python3|sh|bash)
            exec "$@"
            ;;
    esac
fi

case "$role" in
    api)
        exec uvicorn "${APP_MODULE:-app.main:app}" \
            --host "${APP_HOST:-0.0.0.0}" \
            --port "${APP_PORT:-8000}" \
            --workers "${APP_WORKERS:-1}" \
            --loop uvloop \
            --no-access-log \
            "$@"
        ;;
    worker)
        exec celery -A "${CELERY_APP:-app.services.worker}" worker \
            --loglevel="${CELERY_LOG_LEVEL:-info}" \
            --concurrency="${CELERY_CONCURRENCY:-1}" \
            --queues="${CELERY_QUEUES:-cv}" \
            "$@"
        ;;
    *)
        exec "$role" "$@"
        ;;
esac
