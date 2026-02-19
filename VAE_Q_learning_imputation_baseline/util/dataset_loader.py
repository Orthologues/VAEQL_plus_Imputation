#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Desicrption: This file is used to define the class with the methods to load the dataset for the VAE-Q learning imputation baseline.
#########################################################

import pandas as pd
from pyspark.sql import SparkSession
from config import FeaturesTypeDict


def dataset_loader_pandas(dataset_name: str):
    """Load the dataset based on the provided dataset name."""
    if dataset_name == "example_dataset":
        # Load your dataset here, e.g., using pandas or any other library
        # For example:
        # import pandas as pd
        # data = pd.read_csv("path_to_your_dataset.csv")
        # return data
        pass
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")   
    

def dataset_loader_pyspark(dataset_name: str):
    """Load the dataset based on the provided dataset name."""
    if dataset_name == "example_dataset":
        # Load your dataset here, e.g., using pandas or any other library
        # For example:
        # import pandas as pd
        # data = pd.read_csv("path_to_your_dataset.csv")
        # return data
        pass
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")   