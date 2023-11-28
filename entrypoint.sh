#!/bin/bash

curl --location --output artifacts.zip --location --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "https://gitlab.com/api/v4/projects/${PROJECT_ID}/jobs/${MODELS_ARTIFACT_ID}/artifacts"

unzip artifacts.zip

rm -f artifacts.zip

echo "Starting server"

embed-rs