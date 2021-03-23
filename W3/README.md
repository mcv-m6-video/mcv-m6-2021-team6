## Files & fodlers structure

```
mcv-m6-2021-team6
├── datasets
|    ├── AICIty_data
|    |    └── ...
|    ├── frames
|    |    └── ...
|    └── ai_challenge_s03_c010-full_annotation.xml
├── utils
│   └── ...
├── W1
│   └── ...
├── W2
│   └── ...
└── W3
    ├── deep_sort
    |    └── ...
    ├── evaluation
    ├── object_detection
    ├── __several_files.py_
    └── ....
```

## Install requirements
Requirements can be installed from `requirements.txt` with the command: `pip install -r requirements.txt`


## Running Task 1_1, 1_2 & 1_3:

This should be as easy as running: `python task1.py` with two methods in main: task1_1() & task1_2() which does tasks related to 1.2 & 1.3

## Running Task 2_1:

This should be as easy as running: `python task2_1.py`  with a method in main: task2_1()

## Running Task 2_2:

This should be as easy as running: `python task2_2.py --save_path "/home/group01/pycharm_tmp2/output/video.avi"  --ignore_display  "/home/group01/pycharm_tmp2/datasets/AICity_data/train/S03/c010/vdo.avi"` being both parameters (save_path & ignore_display optional) and the last parameter, the path to the video to be processed
