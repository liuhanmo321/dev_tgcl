# Learning Towards the Future on Temporal Graphs

This is the implementation for the paper "Learning Towards the Future on Temporal Graphs".

The codes are in the folder 'SubGraphCL'.

To run the code, please use the command
```
python run.py
```

In run.py, the important parameters are:

- `dataset`: the dataset to use, can choose from 'yelp', 'reddit', 'amazon'.
- `model`: the model to use, can choose from 'TGAT' or 'DyGFormer'.
- `method`: the method to use, 'SubGraphCL' represents our method (LTF in the paper).

The other parameters are the hyper-parameters for the model.

The data are available at [this link](https://figshare.com/s/4cac2181466dfe82b84e).
Please download the data and unzip them in the folder 'SubGraphCL/data'.