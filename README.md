#  Neural content-aware collaborative filtering for cold-start music recommendation

This repository contains the code for reproducing the experiments in our paper entitled [Neural content-aware collaborative filtering for cold-start music recommendation](https://arxiv.org/abs/2102.12369), published in Data Mining and Knowledge Discovery.

## Setup

### Getting the data

After cloning or downloading this repository, you will need to get the data from the [Million Song Dataset](http://millionsongdataset.com/) to reproduce the results.

* The [playcounts](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip) from the Taste Profile set.
* The list of [unique tracks](http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt).

You will also need the pre-computed Statistical Spectrum Descriptors as acoustic data. These are available [online](http://www.ifs.tuwien.ac.at/mir/msd/download.html) (download the file `msd-ssd-v1.0.arff.gz`).

All the files should be unziped (if needed) and placed in the `data/` folder.
Note that you can change the folder structure, as long as you change the path accordingly in the code.

### Requirements

This repository relies on several python packages. For convenience, these are stacked in the `requirements.txt` file. You can install them by simply running:

```
pip3 install requirements.txt
```

## Reproducing the experiments

### Data preprocessing

Now that you're all set, first run the `prep_data.py` script to preprocess the data and create the splits.

### Validation

Then, run the `training_validation.py` script to perform validation over the hyperparameters.
Note that you can run the script using several arguments instead of performing all computations at once.
For instance :

```
python3 training_validation.py -s warm -m mf_hybrid ncf
```

will only perform validation of the MF-Hybrid and NCF models in the warm-start setting.
The list of arguments can be accessed by running `python3 training_validation.py -h`.
By default, all models are trained and validated.

### Testing

The `test.py` script train the models using optimal hyperparameters and compute the NDCG on the test set for all splits.
Similarly to `training_validation.py`, it can be run with several arguments in order to train/test the selected models only.
For instance:

```
python3 training_validation.py -s cold -m ncacf -k 2 3 7
```

will only train and test the NCACF model in the cold-start setting, for splits of indices 2, 3 and 7.

The list of arguments can be accessed by running `python3 test.py -h`.
By default, all models are trained and tested and all splits.

Note that this script uses the hyperparameters obtained after validation.
If you do not wish to validate the models beforehand, it uses pre-defined hyperparameters values.

### Getting the results

To display the results from the papers, simply run the `display_results.py` script, which procudes the validation plots (Fig. 4-8) and several validation and test tables (Tables 4-6).


## References

<details><summary>If you use any of this code for your research, please cite our paper:</summary>
  
```latex
@article{Magron2022,  
  author={P. Magron and C. F{\'e}votte},  
  title={Neural content-aware collaborative filtering for cold-start music recommendation},  
  journal={Data Mining and Knowledge Discovery},  
  year={2022}
}
```

Also note that part of this code is taken from the [content_wmf](https://github.com/dawenl/content_wmf) repository.
Please consider citing the corresponding paper:
  
```latex
@inproceedings{Liang2015,
    author = {Liang, D. and Zhan, M. and Ellis, D.},
    title = {Content-aware collaborative music recommendation using pre-trained neural networks},
    booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
    year = {2015},
    month = {October}
}
```

</p>
</details>
