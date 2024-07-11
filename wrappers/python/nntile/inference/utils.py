import json
from enum import Enum


def dump_pydantic_model(model_object):
    return json.dumps(
        model_object.__dict__,
        indent=4,
        default=lambda x: x.value if isinstance(x, Enum) else x,
    )
