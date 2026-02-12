from engines.crf_engine import CRFChordEngine
from engines.transformer_engine import ARTransformerEngine


def create_engine(engine_type):

    if engine_type == "crf":
        return CRFChordEngine()

    elif engine_type == "transformer":
        return ARTransformerEngine()

    else:
        raise ValueError("Unknown engine type")