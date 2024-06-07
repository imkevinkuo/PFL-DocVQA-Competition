# PFL-DocVQA Competition

Please refer to https://github.com/rubenpt91/PFL-DocVQA-Competition/tree/master for the full instructions.

# Track 1 Submission: LoRA with Quantized Communication (CMU)

## Setup
Follow the instructions in https://github.com/rubenpt91/PFL-DocVQA-Competition/blob/master/framework_documentation/how_to_use.md#download-dataset to set up the workspace.

We additionally require Tensorflow (for logging) and PEFT (for LoRA). These should be install via pip:
```
python -m pip install tensorflow_cpu peft
```

## How to run

Our code adds three new scripts:
- `fl_train.py` is a parallel script that contains most of the code for FL training.
- `fl_utils.py` loads the checkpoints from `fl_train.py`, merges them with the backbone architecture, and then saves it using the competition format.
- `fl_quant.py` contains helper functions for NF4 quantization.

To train LoRA with the default competition hyperparameters (1 round / 2 clients per round), run:

```
python fl_train.py  --name path_to_run_folder --num_rounds 8 --sample_clients 2 --iterations_per_fl_round 1
```

To train LoRA with more efficient communication, we can use more local epochs, a single client per round, and NF4 quantization:

```
python fl_train.py  --name path_to_run_folder --num_rounds 2 --sample_clients 1 --iterations_per_fl_round 16 --quantize
```

By default, we use rank=6, learning rate 2e-4, and batch size 16 (2 for each GPU on 8 GPUs).

To separately run validation on a saved checkpoint, run:

```
python fl_train.py  --eval_ckpt --ckpt /path_to_run_folder/my_checkpoint.ckpt
```

To merge the LoRA adapters and generate the full model checkpoint used for submission, run:
```
python fl_utils.py  --merge_lora --ckpt /path_to_run_folder/my_checkpoint.ckpt
```

The checkpoint is saved as a folder `/path_to_run_folder/my_checkpoint_merged.ckpt`.

```
cd path_to_run_folder
zip my_checkpoint_merged.ckpt.zip my_checkpoint_merged.ckpt -r
```

We then upload the file `my_checkpoint_merged.ckpt.zip` to the ELSA submission server.

## Editing

Our code adds several new arguments in `utils.py` in order to use the original config loading code. See `utils.py` for more details.

Our code uses PyTorch Distributed; see [Writing Distributed Applications with PyTorch
](https://pytorch.org/tutorials/intermediate/dist_tuto.html).

## Reproducing Efficient Results

To reproduce our most efficient results, we use the following commands:

```
python fl_train.py  --name c0 --client_id 0 --iterations_per_fl_round 16
```

If `--client_id` is specified, `fl_train.py` will only train on `--client_id`. Here, we train on client 0 for 16 epochs. The checkpoint is saved in `c0/epoch_15.ckpt`. 

```
python fl_train.py  --name c1 --client_id 1 --ckpt c0/epoch_15.ckpt --quantize --iterations_per_fl_round 16
```

Adding the `--quantize` flag will load a quantized version of `c0/epoch_15.ckpt`. This simulates the quantization that clients perform when uploading their model update. We then train on client 1 for 16 epochs. This new checkpoint is saved in `c1/epoch_15.ckpt`.

*We also allow the user to include `--quantize` in FL training, which will perform this upload quantization automatically. If sampling more than one client per round, the parameters will no longer be quantized after aggregation. Therefore, we quantize the parameters again before download.*


```
python fl_utils.py --ckpt c1/epoch_15.ckpt --merge_lora --quantize
```

`fl_utils.py` saves the final merged checkpoint in a folder called `c1/epoch_15_merged.ckpt`.