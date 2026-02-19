#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the typed dictionaries for the VAE-Q learning imputation baseline (non-cloud baseline solution run at on-premises servers).
#########################################################

from typing import TypedDict, Set, Required

class FeaturesTypeDict(TypedDict, total=False):
    """Dictionary type for dataset features and their types."""
    all_feats: Required[Set[str]] # Set of all feature names
    real_val_feats: Required[Set[str]] # Set of real-valued feature names
    pos_real_val_feats: Set[str] # Set of positive real-valued feature names
    count_feats: Set[str] # Set of count feature names
    ord_feats: Set[str] # Set of ordinal feature names
    cat_feats: Set[str] # Set of categorical feature names
