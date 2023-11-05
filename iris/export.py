import argparse
import os
import time
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from iris.data import IrisLitDataModule
from iris.litmodules import IrisLitModule, get_model

#################### ONNX EXPORT NN MODULE ####################
###############################################################


class IrisONNX(torch.nn.Module):
    """
    Iris torch.nn.Module for exporting models with preprocessing to ONNX format
    """

    def __init__(
        self, lit_module: IrisLitModule, preprocessing: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        if preprocessing is not None:
            self.model = torch.nn.Sequential(preprocessing, lit_module.model)
        else:
            self.model = lit_module.model

    def forward(self, x: torch.Tensor):
        preds = self.model(x)
        try: return preds["out"] # for segmentation models
        except: return preds # for all other models


#################### EXPORT FUNCTION ####################
#########################################################


def export(
    model_root: Optional[str] = f"{os.getenv('AZUREML_MODEL_DIR')}",
    model_arch: Optional[str] = None,
    model_id: Optional[str] = None,
    model_alias: Optional[str] = "best",
    preprocessing: Optional[torch.nn.Module] = None,
    config_fname: str = "iris.json",
):
    """
    Export a LightningModule checkpoint (model.ckpt) to ONNX model (model.onnx).

    Arguments:
        - model_root: (optional) the directory where a checkpoint (model.ckpt) is stored
        - model_arch: (optional) the model architecture
        - model_id: (optional) the wandb model id
        - model_alias: (optional) the wandb model alias,
        - preprocessing: (optional) the preprocessing module to be saved into the ONNX export
        - config_fname: the path to the desired config file,
    """
    # parse the config
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
        model_root = f"../../models/{model_arch}/{model_id}/{model_alias}/model.ckpt"

    # get the lit_module from the checkpoint
    lit_module = get_model(cfg, model_root)

    # create a wrapper module for exporting to ONNX
    model = IrisONNX(lit_module, preprocessing)
    model.eval()

    # dummy input
    dummy_input = torch.randint(low=0, high=256, size=(1, 3, 1080, 1920))

    # define the output path
    ckpt_name = os.path.basename(model_root)
    output_path = model_root.replace(ckpt_name, ckpt_name.replace(".ckpt", ".onnx"))
    # export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        # verbose=True,
        # opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    print(f"Exported model has been saved to {output_path}")

    # validate the ONNX model
    validate_onnx_export(output_path, model)


def validate_onnx_export(
    onnx_model_root: str,
    model: Optional[torch.nn.Module] = None,
):
    onnx_model = onnx.load(onnx_model_root)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_model_root, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def measure_runtime(ort_session, model, dummy_input, n_runs=1):
        pl_rt_list = []
        for _ in range(n_runs):
            # Pytorch Model Runtime
            t0 = time.time()
            dummy_output = model(dummy_input)
            pl_rt_list.append(time.time() - t0)
        pl_rt = np.mean(pl_rt_list)

        onnx_rt_list = []
        for _ in range(n_runs):
            # ONNX Model Runtime
            t0 = time.time()
            ort_dummy_input = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
            ort_dummy_output = ort_session.run(None, ort_dummy_input)
            onnx_rt_list.append(time.time() - t0)
        onnx_rt = np.mean(onnx_rt_list)

        # Compare ONNX and Pytorch Model outputs
        np.testing.assert_allclose(
            to_numpy(dummy_output), ort_dummy_output[0], rtol=1e-03, atol=1e-05
        )

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        print(
            f"Input shape: {dummy_input.shape} \n \
                \t ONNX Runtime: {onnx_rt}s, ~{int(1/onnx_rt)}fps ({int(100 * pl_rt / onnx_rt)}% {'faster' if onnx_rt < pl_rt else 'slower'} than pytorch) \n \
                \t PyTorch Runtime: {pl_rt}s, ~{int(1/pl_rt)}fps"
        )

    # small dummy input (measuring mean runtime)
    measure_runtime(
        ort_session, 
        model,
        torch.randint(low=0, high=256, size=(1, 3, 540, 540))
    )

    # medium dummy input (measuring mean runtime)
    measure_runtime(
        ort_session, 
        model,
        torch.randint(low=0, high=256, size=(1, 3, 1080, 1920))
    )

    # large dummy input (measuring mean runtime)
    measure_runtime(
        ort_session, 
        model,
        torch.randint(low=0, high=256, size=(1, 3, 3000, 4000))
    )
    

    


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
