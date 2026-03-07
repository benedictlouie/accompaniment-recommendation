from engines.crf_engine import CRFChordEngine
from engines.transformer_engine import ARTransformerEngine


def create_engine(engine_type, tempo):

    if engine_type == "crf":
        return CRFChordEngine(tempo)

    elif engine_type == "transformer":
        return ARTransformerEngine(tempo)

    else:
        raise ValueError("Unknown engine type")