
## Set of deep learning models to classify morphological property of Peptide Amphiphile sequence into _high_ or _low_

## Run on `bash` command line terminal on your computer
```bash
git clone https://github.com/stupplab/predict-PA-properties.git  # download the repository
cd predict-PA-properties                                  # go inside the downloded directory
python -m venv env                                        # create virtual envrironment env
source env/bin/activate                                   # activate the env
pip install -r requirements.txt --no-cache-dir            # install required libraries in the env
python main.py --model-path model.tar --predict seqs.csv  # use the model
```
This will create the `seqs_predict.csv` in the same directory. Add your own sequences to `seqs.csv`. Note that `env` should be activated—if not already—using `source env/bin/activate` before running `main.py`. To deactivate the environment, do `deactivate`.


## Model performance
The model is trained using `train.csv` and evaluated on `test.csv`

Details are as below

|  | H-bonds per PA | CO nematic order | Flatness | RCC | RMSF
| Train Set (high - low) | 2187 - 2142 | 2187 - 2141 | 2187 - 2141 | 2187 - 2142 | 2187 - 2141 |
| Test Set (high - low) | 519 - 564 | 518 - 564 | 518 - 564 | 519 - 564 | 518 - 564 |
| True Positive | 473 | 405 | 489 | 459 | 411 |
| False Positive | 34 | 75 | 54 | 79 | 103 |
| True Negative | 530 | 489 | 510 | 485 | 461 |
| False Negative | 46 | 113 | 29 | 60 | 107 |
| Precision (%) | 92 | 84 | 90 | 85 | 80 |
| Accuracy (%) | 91 | 83 | 92 | 87 | 81 |

