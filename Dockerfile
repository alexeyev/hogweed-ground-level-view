FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ARG APT_INSTALL="apt-get install -y --no-install-recommends"

# should be: drwxrwxrwt
# otherwise: Couldn't create temporary file /tmp/apt.conf.XXXXXX for passing config to apt-key
RUN chmod 1777 /tmp
RUN apt-get update && apt-get install -y python3-pip git vim p7zip-full

## Locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

## Libs
RUN python -m  pip install -r requirements.txt

## Data
RUN zenodo_get 5233380
RUN 7z x images.7z && mv images/* prepared_data/images/