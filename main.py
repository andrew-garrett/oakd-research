import argparse

import oak_classification.engines
import oak_detection.engines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for various training and logging vision pipelines",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    )
    parser.add_argument("--task", default="instseg", type=str, help="type of training task for network selection", choices=["cls", "objdet", "instseg"]) # "semseg", 
    # parser.add_argument("--mode", default="val", type=str, help="whether to train or validate a model", choices=["train", "val"])
    ARGS = parser.parse_args()
    if ARGS.task == "cls":
        CFG_FNAME = "./oak_classification/model_cfg.json"
        engine = lambda cfg_fname: oak_classification.engines.base_engine(cfg_fname)
    elif ARGS.task == "objdet":
        CFG_FNAME = "./oak_detection/model_cfg.json"
        engine = lambda cfg_fname: oak_detection.engines.base_engine(cfg_fname)
    elif ARGS.task == "instseg":
        CFG_FNAME = "./oak_segmentation/model_cfg.json"
        engine = lambda cfg_fname: oak_detection.engines.base_engine(cfg_fname)
    else:
        raise NotImplementedError
    
    engine(CFG_FNAME)