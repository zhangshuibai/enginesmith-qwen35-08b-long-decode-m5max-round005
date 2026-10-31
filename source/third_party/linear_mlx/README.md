These files are copied unmodified from MLX 0.32.4, distributed under the MIT
license in LICENSE. The source version is the session's installed mlx 0.32.4.
The steel headers implement Apple's NAX and simdgroup matrix tiles. The rms_norm.metal
file was fetched from the upstream v0.32.4 tag and is retained as provenance.

`tools/gen_linear_nax.py` flattens the headers and appends the engine's specialized
Q4 entry point; the resulting embedded source is `src/kernels/linear_nax.metal`.
This generator is a development utility and is not needed to build or run.
`tools/gen_linear_splitk.py` similarly flattens the ordinary MMA helpers,
removing the unused complex-number specializations, then adds the 32x32
split-K entry points. The original vendor files themselves remain unmodified.

The QMV, RMSNorm, sigmoid and logsumexp reductions in `src/kernels/linear.metal`
are adaptations of MLX 0.32.4's quantized.h, rms_norm.metal, unary_ops.h and
logsumexp.h. Their buffer interfaces and fixed shapes are specialized here.
