from setuptools import find_packages, setup

setup(
    name="iris",
    version="0.1.1",
    description="Open-Source Neural Network Training and Inference Package",
    author="Andrew Garrett",
    author_email="andrewgarrett52@gmail.com",
    url="https://github.com/andrew-garrett/oakd-research",
    install_requires=[
        # "lightning",
        # "torch",
        # "opencv-contrib-python",
        # "scikit-learn", # ----------------------> classify.py only
        # "matplotlib", # ---------------------------------------------------> segment.py only
        # "xgboost", # ---------------------------> classify.py only
        # "pandas", # ---------------> data.py only
        # "segment_anything", # ---------------------------------------------> segment.py only
        # "numpy",
        # "scipy", # -----------------------------> classify.py only
        # "torchvision",
        # "wandb",
    ],
    packages=find_packages() + ["iris"],
    package_data={
        "iris": ["configs/*.json"]
    },  # Includes all .json files in the configs/ directory
)
