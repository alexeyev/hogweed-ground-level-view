# Detecting Hogweed on the Ground-Level View Photographs: Dataset

Hogweed (Heracleum) is a herbs genus that features many invasive species such as giant hogweed 
or Sosnowsky's hogweed. This invasive species are particularly notorious due to the high 
content of phototoxic compounds, so that any contact with a plant may result in an intense skin burn. 

Invasion of the [Sosnowsky's hogweed](https://antiborschevik.info/) \[lang:RU\] in particular is major trouble 
in Central Russia, and by 2021 resolving the problem requires massive intervention. Agtech drones spraying 
herbicides are already used to eradicate the Sosnowsky's hogweed, and accompanying real-time detection 
algorithms for UAVs are being developed (e.g. see [this paper](https://ieeexplore.ieee.org/document/9359491) 
and [the related dataset repository](https://github.com/DLopatkin/Heracleum-Dataset)).

We propose a dataset for detecting Sosnowsky's hogweed using the ground-level view as if we're 
looking through the camera of an **autonomous unmanned ground vehicle** patrolling the hogweed-endangered 
area (e.g. a week after mowing or poisoning). It is not 100% clear whether this dataset can or should be 
used for training actual robotic vision algorithms. However, plant detection in the natural environment 
is quite a challenge, which makes such annotated images collections suitable for competitions 
and/or ML homeworks. This is a *grassroot* (pun intended) initiative without any external funding (so far).

## Data

**444** (311/133) photos are taken in different locations in Russia using a Samsung Galaxy A31 camera. 
The images are annotated using https://supervise.ly/ (CE). 

Data for `prepared_data/images/` (CC-BY-4.0) can be downloaded from Zenodo: [5233380](https://zenodo.org/record/5233380).

A more detailed description of the data collection strategy and the dataset in general will be released during autumn.
Test set annotations will be released in December, by the end of the autumn semester.

## Format

The annotations are provided in COCO format. To inspect the annotations manually, please see 
the jupyter notebook `COCO-formatted-annotations-viewer.ipynb` adapted from the [original Gist](https://gist.github.com/akTwelve/dc79fc8b9ae66828e7c7f648049bc42d) 
shared by [akTwelve](https://github.com/akTwelve).

## How to cite

We would appreciate if you cite this dataset as

```
@dataset{alekseev_anton_2021_5233380,
  author       = {Alekseev, Anton},
  title        = {{Detecting Hogweed on the Ground-Level View Photographs: Dataset}},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5233380},
  url          = {https://doi.org/10.5281/zenodo.5233380}
}
```

## Acknowledgements

I would like to thank Aleksey Artamonov and Mikhail Evtikhiev for various consultations and proofreading.

## Other materials

* [A monster that devours Russia](https://www.youtube.com/watch?v=u5NxuEoXHn8) \[YouTube video\]
* Different species, similar threat: [Giant Hogweed - The UK's Most Dangerous & Toxic Plant](https://www.youtube.com/watch?v=p2iCSHrYjoc) \[YouTube video, possibly disturbing content\]


![Semantic segmentation](example_coco_annotation.jpg?raw=true "Polygons obtained via manual annotation.")
