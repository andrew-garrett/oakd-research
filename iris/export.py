import argparse
import os
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from iris.data import IrisLitDataModule
from iris.litmodules import get_model

#################### EXPORT FUNCTION ####################
#########################################################


def export(
    model_root: Optional[str] = f"{os.getenv('AZUREML_MODEL_DIR')}",
    model_arch: Optional[str] = None,
    model_id: Optional[str] = None,
    model_alias: Optional[str] = "best",
    config_fname: str = "iris.json",
):
    """
    Export a LightningModule checkpoint (model.ckpt) to ONNX model (model.onnx).

    Arguments:
        - model_root: (optional) the directory where a checkpoint (model.ckpt) is stored
        - model_arch: (optional) the model architecture
        - model_id: (optional) the wandb model id
        - model_alias: (optional) the wandb model alias,
        - data_root: the directory where a dataset is stored or the path to a json file,
        - config_fname: the path to the desired config file,
    """
    cfg = IrisLitDataModule.parse_config(config_fname)
    # update the number of classes using ignore indices
    if "num_classes" in cfg.keys() and "ignore_index" in cfg.keys():
        ignore_index = cfg["ignore_index"]
        if type(ignore_index) != list:
            ignore_index = [ignore_index]
        ignore_index = [i for i in ignore_index if (i >= 0 and i <= cfg["num_classes"])]
        cfg["num_classes"] -= len(ignore_index)
    # if the model_root arg is not provided, we build the path
    if model_root is None:
        model_root = f"../../models/{model_arch}/{model_id}/{model_alias}"
    litmodule = get_model(cfg, model_root)
    litmodule.eval()
    model = litmodule.model
    dummy_input = litmodule.example_input_array
    dummy_output = litmodule(dummy_input)
    ckpt_name = os.path.basename(model_root)
    output_path = model_root.replace(ckpt_name, ckpt_name.replace(".ckpt", ".onnx"))
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        output_path,
        # opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    validate_onnx_model(output_path, dummy_input, dummy_output)


def validate_onnx_model(
    model_root: str, dummy_input: torch.Tensor, dummy_output: torch.Tensor
):
    onnx_model = onnx.load(model_root)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(model_root, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05
    )

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="export.py",
        description="ONNX Export script for iris",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=str,
        help="The filepath for the model checkpoint, must to be .ckpt. \
            if not specifed, must provide a valid pairing of [--model-arch, --model-id, --model-alias] arguments",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        help="The desired model architecture",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="The wandb run ID for the model checkpoint",
    )
    parser.add_argument(
        "--model-alias",
        default="best",
        type=str,
        choices=["best", "latest", "v0"],
        help="The wandb artifact alias for the model checkpoint",
    )
    # parser.add_argument(
    #     "--data-root",
    #     default="./tmp/",
    #     type=str,
    #     help="The root where the dataset folder is located",
    # )
    # parser.add_argument(
    #     "--n-gpus",
    #     default=torch.cuda.device_count(),
    #     type=int,
    #     help="Number of GPUs, 0 means cpu, 1 means single gpu, >1 means distributed",
    # )
    ARGS = parser.parse_args()

    # get the model from checkpoint and export it to ONNX format
    export(
        model_root=ARGS.model_root,
        model_arch=ARGS.model_arch,
        model_id=ARGS.model_id,
        model_alias=ARGS.model_alias,
    )
