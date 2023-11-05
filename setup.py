from setuptools import find_packages, setup

setup(
    name="iris",
    version="0.0.1",
    description="Open-Source Neural Network Training and Inference Package",
    author="Andrew Garrett",
    author_email="andrewgarrett52@gmail.com",
    url="https://github.com/andrew-garrett/oakd-research",
    install_requires=[
        # "lightning",
        # "torch",
        # "numpy",
        # "torchvision",
        # "wandb",
        # "onnx",
        # "onnxruntime",
        # "azureml-inference-server-http"
    ],
    packages=find_packages() + ["iris"],
    package_data={"iris": ["configs/*.json"]},
)
