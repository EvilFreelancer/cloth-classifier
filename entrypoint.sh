#!/bin/bash

cd "$(dirname "$0")"

[ "x$APP_WORKERS" = "x" ] && export APP_WORKERS="1"
[ "x$APP_BIND" = "x" ] && export APP_BIND="0.0.0.0"
[ "x$APP_PORT" = "x" ] && export APP_PORT="8080"

gunicorn --workers $APP_WORKERS --bind ${APP_BIND}:${APP_PORT} wsgi:app
