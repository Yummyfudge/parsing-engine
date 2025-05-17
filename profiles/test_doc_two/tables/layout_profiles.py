


from .layout_hooks import (
    keep_tables_only,
    construct_labelled_block
)

layout_profiles = {
    "default": {
        "filter_fn": keep_tables_only,
        "constructor_fn": construct_labelled_block
    },
    "tables_only": {
        "filter_fn": keep_tables_only,
        "constructor_fn": construct_labelled_block
    },
    "debug_all": {
        "filter_fn": lambda label, score: True,
        "constructor_fn": construct_labelled_block
    }
}