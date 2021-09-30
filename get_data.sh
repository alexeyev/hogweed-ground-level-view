#!/usr/bin/env bash

echo "Loading images from Zenodo..."
zenodo_get 5233380

echo "Unpacking images"
7z x images.7z && mv images/* prepared_data/images/

echo "Splitting into folders"
python3 utils/prepare_data_for_classification.py