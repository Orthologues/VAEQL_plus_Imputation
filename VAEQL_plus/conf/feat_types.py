#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the typed dictionaries for the VAEQL imputation pipeline.
#########################################################

from typing import TypedDict, Set, Required, Dict
import json

class FeaturesTypeDict(TypedDict, total=False):
    
    """Dictionary type for dataset features and their types."""
    all_feats: Required[Set[str]] # Set of all feature names
    real_val_feats: Set[str] # Set of real-valued feature names
    pos_real_val_feats: Required[Set[str]] # Set of positive real-valued feature names
    count_feats: Set[str] # Set of count feature names
    ord_feats: Dict[str, int] # Dictionary of ordinal feature names, keys are feature names and values are the number of orders
    bi_feats: Dict[str, Set[str]] # Dictionary of binary feature names, keys are feature names and values are the two possible categories
    cat_feats: Dict[str, Set[str]] # Dictionary of categorical feature names, keys are feature names and values are the possible categories


    """
    Usage: features = FeaturesTypeDict.create("icu_history.json") will raise on any mismatch and otherwise return the validated dict
    * cls: same as "self" for class methods, refers to the class itself (FeaturesTypeDict)
    """
    @classmethod
    def create(cls, input_json_path: str) -> "FeaturesTypeDict":
          
        with open(input_json_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        # required keys
        for key in ("all_feats", "pos_real_val_feats"):
            if key not in json_obj:
                raise KeyError(f"Missing required key: {key!r}")

        def to_str_set(name: str) -> Set[str]:
            val = json_obj.get(name, [])
            if not isinstance(val, (list, set, tuple)):
                raise TypeError(f"{name!r} must be a list/iterable of strings")
            if not all(isinstance(x, str) for x in val):
                raise TypeError(f"{name!r} elements must all be strings")
            return set(val)

        # non one-hot-/unary-encoded features
        all_feats = to_str_set("all_feats")
        real_val_feats = to_str_set("real_val_feats")
        pos_real_val_feats = to_str_set("pos_real_val_feats")
        count_feats = to_str_set("count_feats")

        # unary-encoded features
        ord_feats_raw = json_obj.get("ord_feats", dict())
        if not isinstance(ord_feats_raw, dict):
            raise TypeError("ord_feats must be a dict[str, int]")
        ord_feats: Dict[str, int] = {}        
        for k, v in ord_feats_raw.items():
            if not isinstance(k, str) or not isinstance(v, int):
                raise TypeError("ord_feats keys must be strings and values ints")
            if v < 2:
                raise ValueError(f"ord_feats[{k!r}] must be >= 2")
            ord_feats[k] = v

        # binary features (single-column 0/1 normalization target)
        bi_feats_raw = json_obj.get("bi_feats", dict())
        if not isinstance(bi_feats_raw, dict):
            raise TypeError("bi_feats must be a dict[str, iterable[str]]")
        bi_feats: Dict[str, Set[str]] = {}
        for k, v in bi_feats_raw.items():
            if not isinstance(k, str):
                raise TypeError("bi_feats keys must be strings")
            if not isinstance(v, (list, set, tuple)):
                raise TypeError(f"bi_feats[{k!r}] must be an iterable of strings")
            if len(v) != 2:
                raise ValueError(f"bi_feats[{k!r}] must have exactly two categories")
            if not all(isinstance(x, str) for x in v):
                raise TypeError(f"bi_feats[{k!r}]'s elements must be strings")
            bi_feats[k] = set(v)
            
        # one-hot-encoded features
        cat_feats_raw = json_obj.get("cat_feats", dict())
        if not isinstance(cat_feats_raw, dict):
            raise TypeError("cat_feats must be a dict[str, iterable[str]]")
        cat_feats: Dict[str, Set[str]] = {}
        for k, v in cat_feats_raw.items():
            if not isinstance(k, str):
                raise TypeError("cat_feats keys must be strings")
            if not isinstance(v, (list, set, tuple)):
                raise TypeError(f"cat_feats[{k!r}] must be an iterable of strings")
            if len(v) < 3:
                raise ValueError(
                    f"cat_feats[{k!r}] must have at least three categories; use bi_feats for binary features"
                )
            if not all(isinstance(x, str) for x in v):
                raise TypeError(f"cat_feats[{k!r}]'s elements must be strings")
            cat_feats[k] = set(v)

        # union check
        union_feats = set().union(
            real_val_feats,
            pos_real_val_feats,
            count_feats,
            set(ord_feats.keys()),
            set(bi_feats.keys()),
            set(cat_feats.keys()),
        )
        if union_feats != all_feats:
            raise ValueError(
                f"Union of feature subsets (len={len(union_feats)}) does not match all_feats (len={len(all_feats)})"
            )
        

        # return to the created value loaded from a .json file
        return cls(
            all_feats=all_feats,
            real_val_feats=real_val_feats,
            pos_real_val_feats=pos_real_val_feats,
            count_feats=count_feats,
            ord_feats=ord_feats,
            bi_feats=bi_feats,
            cat_feats=cat_feats,
        )
          
