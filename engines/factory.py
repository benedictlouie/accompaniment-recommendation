from engines.crf_engine import CRFChordEngine
from engines.transformer_engine import ARTransformerEngine


def create_engine(engine_type, tempo):

    if engine_type == "crf":
        return CRFChordEngine(tempo)

    elif engine_type == "transformer":
        return ARTransformerEngine(tempo)

    elif engine_type == "onnx":
        from engines.onnx_engine import ONNXChordEngine
        return ONNXChordEngine(tempo)

    else:
        raise ValueError("Unknown engine type")