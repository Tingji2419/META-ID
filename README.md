To reproduce the experiments, please first install the necessary packages:

```
pip install -r requirements.txt
```

and then run:

```
bash generate_dataset-meta.sh

cd command-final

bash Beauty_t5_metapath_linear.sh
```

Note: this might takes several hours for training one time on 4 NVIDIA-3090 GPUs.