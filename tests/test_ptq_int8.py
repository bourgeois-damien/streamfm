from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from sgmse.backbones.streaming_unet import CausalConv2d
from sgmse.util.ptq_int8 import (
    QuantizedCausalConv2d,
    QuantizedFallbackWrapper,
    apply_ptq_int8_,
    parse_ptq_components,
    set_quant_engine,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 4
        self.input_freqs = 16
        self.lin = nn.Linear(8, 8)
        self.conv = nn.Conv2d(4, 4, kernel_size=1)
        self.causal = CausalConv2d(4, 4, kernel_size=(3, 3))

    def forward(self, x, time_cond=None):
        _ = time_cond
        h = self.causal(x)
        h = self.conv(h)
        # Exercise the Linear without changing spatial shape of h.
        _ = self.lin(torch.zeros(x.shape[0], 8, device=x.device, dtype=x.dtype))
        return h


class PtqInt8Test(unittest.TestCase):
    def test_fp16_cast_warning_then_ptq(self) -> None:
        set_quant_engine()
        model = TinyNet().eval().half()
        quantized = apply_ptq_int8_(model, "linear", calib_steps=1)
        self.assertTrue(
            any(type(m).__module__.startswith("torch.ao.nn.quantized.dynamic") for m in quantized.modules())
        )
        self.assertEqual(parse_ptq_components("all"), ("linear", "conv", "causal_conv"))
        self.assertEqual(parse_ptq_components("causal_conv,linear"), ("linear", "causal_conv"))
        self.assertEqual(parse_ptq_components("none"), ())
        with self.assertRaises(ValueError):
            parse_ptq_components("fp8")

    def test_dynamic_linear_and_static_convs(self) -> None:
        set_quant_engine()
        model = TinyNet().eval()
        quantized = apply_ptq_int8_(model, "all", calib_steps=2)
        self.assertTrue(
            any(
                type(m).__module__.startswith("torch.ao.nn.quantized.dynamic")
                for m in quantized.modules()
            )
        )
        self.assertTrue(any(isinstance(m, QuantizedCausalConv2d) for m in quantized.modules()))
        qcausal = next(m for m in quantized.modules() if isinstance(m, QuantizedCausalConv2d))
        self.assertTrue(
            type(qcausal.qconv).__module__.startswith("torch.ao.nn.quantized"),
            msg=f"expected quantized conv, got {type(qcausal.qconv)}",
        )
        weight = qcausal.qconv.weight() if callable(qcausal.qconv.weight) else qcausal.qconv.weight
        self.assertTrue(getattr(weight, "is_quantized", False), msg=f"weight not quantized: {weight}")

        x = torch.randn(1, 4, 16, 4)
        with torch.inference_mode():
            y = quantized(x, time_cond=torch.tensor([0.5]))
        self.assertEqual(tuple(y.shape), (1, 4, 16, 4))

    def test_fp16_fallback_keeps_unquantized_modules_and_state_in_fp16(self) -> None:
        set_quant_engine()
        model = TinyNet().eval().half()
        quantized = apply_ptq_int8_(
            model,
            "all",
            calib_steps=2,
            fallback_dtype=torch.float16,
        )

        self.assertEqual(quantized.conv.fallback_dtype, torch.float16)
        self.assertIsInstance(quantized.conv, QuantizedFallbackWrapper)
        self.assertIsInstance(quantized.lin, QuantizedFallbackWrapper)
        self.assertIsInstance(quantized.causal, QuantizedCausalConv2d)
        self.assertEqual(quantized.causal.fallback_dtype, torch.float16)
        self.assertEqual(quantized._streamfm_ptq_int8["fallback_dtype"], "float16")

        x = torch.randn(1, 4, 16, 4, dtype=torch.float16)
        with torch.inference_mode():
            y = quantized(x, time_cond=torch.tensor([0.5], dtype=torch.float16))
            state = quantized.causal.init_state(input_freqs=16)
            step, next_state = quantized.causal.forward_step(x[..., :1], state=state)

        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(step.dtype, torch.float16)
        self.assertEqual(state[0].dtype, torch.float16)
        self.assertEqual(next_state[0].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
