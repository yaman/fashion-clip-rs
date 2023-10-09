#!/bin/bash

ls -althr
pwd
# install dev requirements
pip install --no-cache-dir -r ./.devcontainer/requirements.txt

cd /tmp

# install docker to interact with the host if you plan to use remote docker inside the devcontainer
# for example to train a model on the host but in a container, too
# curl -fsSL https://get.docker.com -o get-docker.sh
# sh get-docker.sh

# install aws cli to interact with the aws cli from the devcontainer
# don't forget to configure the aws cli with your credentials inside devcontainer.json
# apt-get install zip -y
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# ./aws/install