#!/usr/bin/env bash

echo "Loading images from Zenodo..."
zenodo_get 5233380

echo "Unpacking images"
7z x images.7z && mv images/* prepared_data/images/

echo "Splitting into folders"
python3 utils/prepare_data_for_classification.py
python3 prepare_resized_data.py --size 300 --segment train
python3 prepare_resized_data.py --size 300 --segment test