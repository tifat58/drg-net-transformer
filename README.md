# drg-net-transformer
## Installation
Recommended virtual enviroment with python 3.7

To install the dependencies create a virtualenv and run:
```shell
$ pip install -r requirements.txt
```

## Download dataset
- FGADR: https://drive.google.com/drive/folders/1WwSyXfpf0uZjou0CBSIXc7RE_1L8nX0C?usp=sharing

### Update path variables
- In train_fgadr.py uder 'Basic settings'
```
"""Basic Setting"""
data_path = r'path-to-fgadr-dataset' e.g. /home/dataset/FGADR/Original_Images
csv_path = r'path-to-csv-file' e.g. /home/dataset/FGADR/Grading_Label.csv
save_model_path = r'directory-path-to-save-model' 
```


** Run **
```shell
python train_fgadr.py
```
```shell
python train_fgadr_gain.py
```
