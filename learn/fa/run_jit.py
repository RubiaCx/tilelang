import os
import sys

# Ensure we can import the in-tree TVM and TileLang
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TVM_PYTHON = os.path.join(ROOT, "3rdparty", "tvm", "python")
for p in (ROOT, TVM_PYTHON):
    if p not in sys.path:
        sys.path.insert(0, p)

from tvm import ir as I, tir  # type: ignore
from tvm.target import Target  # type: ignore
import tilelang
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.learn.fa.ws_test_stage_3 import Module  # 你的 TIR 文件

target = Target("cuda -arch=sm_90")

mod = Module  # 这是一个 IRModule
# 如果你是从已经很后面的 TIR 开始，可以直接跳过 LowerAndLegalize
# 否则可以视情况先跑 LowerAndLegalize
# mod = LowerAndLegalize(mod, target)

with tilelang.transform.PassContext(
    config={"tl.disable_multi_version_buffer": True},
):
    mod = OptimizeForTarget(mod, target)

rt_mod = tir.build(mod, target=target)  # 或 tvm.build(mod, target=target)