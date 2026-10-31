#!/usr/bin/env python3
"""Compare actual native GDN kernels with installed MLX, stage by stage.

Run using a development Python that already has mlx and mlx-lm installed:
    python tools/probe_gdn.py --rows 1,2,19,2048
Fixtures stay under ignored target/gdn-probe; this is not a runtime dependency.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import precise_swiglu
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk


def bits(array):
    mx.eval(array)
    return np.array(array.view(mx.uint16 if array.dtype == mx.bfloat16 else mx.uint32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,19,2048")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["cargo", "build", "--release", "--example", "gdn_probe"], cwd=root, check=True)
    records = []
    with open("/tmp/engine-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rows in map(int, args.rows.split(",")):
            mx.random.seed(29 + rows)
            folder = root / "target" / "gdn-probe" / str(rows)
            folder.mkdir(parents=True, exist_ok=True)
            bf = lambda shape: mx.random.normal(shape).astype(mx.bfloat16)
            fixtures = {
                "raw": bf((1, rows, 6144)),
                "history": bf((1, 3, 6144)),
                "conv_weight": (mx.random.normal((6144, 4, 1))*0.2).astype(mx.bfloat16),
                "a": bf((1, rows, 16)), "b": bf((1, rows, 16)),
                "alog": mx.random.uniform(-2, 2, (16,)), "dt": bf((16,)),
                "state": mx.random.normal((1, 16, 128, 128))*0.1,
                "z": bf((1, rows, 16, 128)), "norm": bf((128,)),
            }
            for name, value in fixtures.items():
                (folder / f"{name}.bin").write_bytes(bits(value).tobytes())
            x = fixtures
            all_x = mx.concatenate([x["history"],x["raw"]], axis=1)
            conv = nn.silu(mx.conv1d(all_x,x["conv_weight"],groups=6144))
            q,k,v = [part.reshape(1,rows,16,128) for part in mx.split(conv,[2048,4096],axis=-1)]
            q,k = normalize_qk(q,k,inv_scale=128**-0.5,eps=1e-6)
            g = compute_g(x["alog"],x["a"],x["dt"])
            beta = mx.sigmoid(x["b"])
            y,state = gated_delta_kernel(q,k,v,g,beta,x["state"])
            out = precise_swiglu(y,x["z"],mx.fast.rms_norm(y,x["norm"],1e-6))
            expected = {
                "conv":conv, "history":all_x[:,-3:,:],
                "qkv":mx.concatenate([q.reshape(1,rows,2048),k.reshape(1,rows,2048),v.reshape(1,rows,2048)],axis=-1),
                "g":g,"beta":beta,"state":state,"y":y,"out":out,
            }
            mx.eval(*expected.values())
            subprocess.run([str(root/"target/release/examples/gdn_probe"),str(folder),str(rows)], cwd=root, check=True)
            for name, value in expected.items():
                expected_bits = bits(value).ravel()
                actual = np.fromfile(folder/f"native_{name}.bin",dtype=expected_bits.dtype)
                count = int(np.count_nonzero(actual != expected_bits))
                record = {"rows":rows,"stage":name,"different":count,"elements":len(actual)}
                records.append(record)
                print(json.dumps(record),flush=True)
    output = root/"target/gdn-probe/results.json"
    output.write_text(json.dumps(records,indent=2)+"\n")
    return int(any(r["different"] for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
