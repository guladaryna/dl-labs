"""Підготовка графа, статична квантизація (пункт 2) і діапазони каналів ваг (пункт 4)."""

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


class ImageReader(CalibrationDataReader):
    """Подає калібрувальні зображення по одному, кожне форми (1, 3, H, W)."""

    def __init__(self, images):
        self.images = images
        self.position = 0

    def get_next(self):
        if self.position >= len(self.images):
            return None
        x = self.images[self.position:self.position + 1]  # зріз зберігає вісь пакета: (1, 3, H, W)
        self.position += 1
        return {"input": x}


def prepare(fp32_path, prep_path):
    """quant_pre_process(fp32, prep, skip_symbolic_shape=True) — рівно як у таблиці."""
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)


def quantize(prep_path, out_path, calib_images, per_channel):
    """quantize_static з параметрами таблиці; per_channel=False — лише для досліду пункту 4."""
    quantize_static(
        str(prep_path),
        str(out_path),
        ImageReader(calib_images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )


def channel_ranges(fp32_path):
    """Для кожного Conv: alpha_c = max|w| за вихідним каналом c, r = max_c alpha_c / median_c alpha_c.

    Ваги беруться з graph.initializer за іменем другого входу вузла, форма (C_out, C_in, kH, kW).
    """
    graph = onnx.load(str(fp32_path)).graph
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    result = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        w = weights[node.input[1]]
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)  # (C_out,)
        r = float(alpha.max() / np.median(alpha))
        result.append({"name": node.name, "shape": list(w.shape), "r": r, "log2_r": float(np.log2(r))})
    return result
