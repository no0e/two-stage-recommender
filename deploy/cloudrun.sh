#!/usr/bin/env bash
# Build the API image with Cloud Build and deploy it to Cloud Run.
#
#   gcloud auth login && gcloud config set project <PROJECT_ID>
#   CATEGORY=Video_Games ./deploy/cloudrun.sh
#
# The artifacts are baked into the image (Cloud Run has no volume to mount),
# and the feature store is in memory. To use Redis instead (Memorystore behind
# a VPC connector, or any managed Redis), pass REDIS_URL in --set-env-vars.
#
# Sized to stay inside the free tier and to cap the bill: one instance at
# most, scaled to zero when idle, so an idle demo costs nothing.
set -euo pipefail

CATEGORY="${CATEGORY:-Video_Games}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-recsys-api}"
PROJECT="$(gcloud config get-value project)"
REPO="${REPO:-recsys}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD)"

test -f "serving_artifacts/${CATEGORY}/meta.json" || {
  echo "No artifacts for ${CATEGORY}: run scripts/export_serving.py first." >&2
  exit 1
}

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" \
  >/dev/null 2>&1 || gcloud artifacts repositories create "${REPO}" \
  --repository-format docker --location "${REGION}"

gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE},_ARTIFACTS=serving_artifacts/${CATEGORY}" .

gcloud run deploy "${SERVICE}" --image "${IMAGE}" --region "${REGION}" \
  --cpu 1 --memory 1Gi --concurrency 20 --min-instances 0 --max-instances 1 \
  --set-env-vars "WORKERS=1" --allow-unauthenticated

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --format 'value(status.url)')"
echo "Deployed: ${URL}"
curl -s "${URL}/health"; echo
curl -s -X POST "${URL}/recommend" -H 'content-type: application/json' \
  -d '{"user_id": 12, "k": 5}'; echo
