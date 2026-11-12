#!/bin/bash
echo "Deploying to RPi 1..."
rsync -avz --exclude '.git' --exclude 'results' --exclude 'venv' --exclude '__pycache__' ./ eldemaster@192.168.1.147:~/fedpdm_client/
echo "Deploying to RPi 2..."
rsync -avz --exclude '.git' --exclude 'results' --exclude 'venv' --exclude '__pycache__' ./ eldemaster@192.168.1.115:~/fedpdm_client/
echo "Deployment complete."
