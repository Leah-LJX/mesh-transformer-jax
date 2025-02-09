import argparse
import json
import time

import jax
import numpy as np
import optax
import os

import wandb
from tqdm import tqdm


from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs
from smart_open import open
from google.cloud import storage
from google.cloud.exceptions import NotFound

from mesh_transformer.util import clip_by_global_norm, additive_weight_decay


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gpt-j-6b-algorithm-344208-720bc02923a4.json'

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
    To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    Modify the config file:
        - set `model_dir` to where the checkpoints should be written during training
        - set `train_set`, `val_set` to index files for your data
        - set `tpu_size` to 8 (if on a v3-8)
        - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
        - the global step will reset to 0, keep that in mind when writing your lr schedule
        - set `name` to specify the name of the Weights & Biases run
        - set `wandb_project` to specify the Weights & Biases project to log to
    To prepare data in the expected data format:
        - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
        - upload the .tfrecords files to GCS
        - save their GCS paths to a index file under `data/`, see existing files for examples
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-model-path", type=str, default=None, help="Base model to finetune")
    parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")

    args = parser.parse_args()
    return args


def save(network, step, bucket, path, mp, aux=None, keep_n=3, delete_old=True):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    try:
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)


def train_step(network, data):
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }

    loss, last_loss, grad_norm, grad_norm_micro = network.train(inputs)

    return (
        np.array(loss).mean(),
        np.array(last_loss).mean(),
        np.array(grad_norm).mean(),
        np.array(grad_norm_micro).mean(),
    )


def eval_step(network, data):
    inputs = {
        "obs": data[:, :-1],
        "target": data[:, 1:],
    }

    out = network.eval(inputs)
    loss = out["loss"]

    return np.array(loss).mean()


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]
   
    # alpha parameter for the exponential moving averages used to compute B_simple
    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

    scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(scheduler)
    )

    params["optimizer"] = opt

    start = time.time()
    tpu_size = jax.device_count()
    if tpu_size < cores_per_replica:
        msg = f"each shard needs a separate device, but device count ({tpu_size}) < shard count ({cores_per_replica})"
        raise ValueError(msg)
    print(f"jax devices: {tpu_size}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    # pick initial ckpt - based on tuning vs train from scratch

    step = 0
    initial_ckpt_state_path = None
    train_loader = None

    if args.tune_model_path:
        print('`--tune_model_path` passed: we are beginning a fine-tuning run')
        fine_tuning = True
        initial_ckpt_state_path = args.tune_model_path
    else:
        print('`--tune_model_path` not passed: we are continuing a fine-tuning run from a checkpoint (or we are not fine-tuning)')
        fine_tuning = False
        initial_ckpt_model_dir = model_dir
        initial_ckpt_path = f"gs://{bucket}/{initial_ckpt_model_dir}"
        meta_path = f"{initial_ckpt_path}/meta.json"

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ckpt_step = meta["checkpoints"][-1]
            initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
            print(f"state will be restored from checkpoint {ckpt_step}")

            step = ckpt_step
            train_loader = meta['aux'][str(ckpt_step)].get("train_loader", None)
        except NotFound:
            # no checkpoint, start at zero
            print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")

    if initial_ckpt_state_path:
        print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    else:
        print("not loading from a checkpoint")

    # set up datasets
    print("setting up datasets")

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps,
                                          per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_loader)

    global_val_batch = per_replica_batch * tpu_size // cores_per_replica

    val_sets = {}

    for k, v in params["val_set"].items():
        val_sets[k] = TFRecordNewInputs(
            f"data/{v}", batch_size=(global_val_batch,), sample_size=seq
        )

    # tok/sec metrics
    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * sequences_per_step

    # load + run
    #ljx add
    print('begin testing....')
    assert model_dir
    client = storage.Client()
    print(dir(client))
    
    bucket='codecontest-bucket'
    path = 'test_result'
    filename = '1'
    content = '123456'
    # create metadata file
    with open(f"gs://{bucket}/{path}/{filename}", "a+") as f:
        f.write(content)

#     try:
#         with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
#             meta = json.load(f)
#     except:
#         # create metadata file
#         with open(f"gs://{bucket}/{model_dir}/meta.json", "w") as f:
#             json.dump({
#                 "step": 0,
#                 "checkpoints": [],
#                 "aux": {}
#             }, f)
#     print('end testing...')
    ###

