#!/usr/bin/env bash
# Install the Google Cloud CLI in a Linux environment without it (an Onyxia
# service, a fresh VM), user-local, no root needed. Then:
#   gcloud auth login --no-launch-browser
#   gcloud config set project <PROJECT_ID>
#   CATEGORY=Video_Games ./deploy/cloudrun.sh
set -euo pipefail
if command -v gcloud >/dev/null; then gcloud --version | head -1; exit 0; fi
cd "${HOME}"
curl -sSLo gcloud.tar.gz \
  https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xzf gcloud.tar.gz && rm gcloud.tar.gz
./google-cloud-sdk/install.sh --quiet --path-update true --usage-reporting false
echo 'Open a new terminal (or: source ~/.bashrc), then run gcloud auth login --no-launch-browser'
