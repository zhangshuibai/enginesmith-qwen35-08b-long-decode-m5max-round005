#!/usr/bin/env python3
"""Regenerate bundled attention kernels from installed MLX 0.32.4 headers.

Usage: python3 attention_generate.py /path/to/mlx/include /path/to/MLX/LICENSE
This is a development utility, never invoked by the build or executable.
"""
from pathlib import Path
import re
import sys

include_root = Path(sys.argv[1])
license_text = Path(sys.argv[2]).read_text()
output = Path(__file__).parent
license_comment = '/*\nAdapted from MLX 0.32.4. Original sources and modifications are recorded in\nATTENTION_INTERFACE.md.\n\n' + license_text + '\n*/\n'

def flatten(name, seen):
    if name in seen:
        return ''
    seen.add(name)
    source = (include_root / name).read_text()
    source = re.sub(r'^#pragma once\s*$', '', source, flags=re.M)
    source = re.sub(r'^#include "([^"]+)"\s*$', lambda m: flatten(m[1], seen), source, flags=re.M)
    return '\n// Begin ' + name + '\n' + source + '\n// End ' + name + '\n'

prefix = '''#include <metal_stdlib>
using namespace metal;
#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

template <typename T> struct Limits;
template <> struct Limits<float> { static constexpr constant float finite_min = -3.4028234663852886e38f; };
template <> struct Limits<bfloat> { static constexpr constant float finite_min = -3.3895313892515355e38f; };
'''
source = flatten('mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h', set())
# Source has both the ordinary and wide-head split kernels. Keep both definitions;
# instantiate only the reference's wide-head split shape for this model.
for name, value in [('align_Q','false'), ('align_K','false'), ('has_mask','false'), ('do_causal','true'), ('has_sinks','false'), ('save_lse','false')]:
    source = re.sub(r'constant bool ' + name + r' \[\[function_constant\(\d+\)\]\];', 'constant bool ' + name + ' = ' + value + ';', source)
source = re.sub(r', function_constant\(\w+\)', '', source)
source += '''\n// Qwen3.5 full attention: BQ=64, BK=32, D=256, WM=4, WN=2.
template [[host_name("attn_prefill")]] [[kernel]]
decltype(attention_nax_dsplit<bfloat, 64, 32, 256, 4, 2>)
attention_nax_dsplit<bfloat, 64, 32, 256, 4, 2>;
'''
(output / 'attention_prefill.metal').write_text(license_comment + prefix + source)

source = (include_root / 'mlx/backend/metal/kernels/sdpa_vector.h').read_text()
for name, value in [('has_mask','false'), ('query_transposed','true'), ('do_causal','false'), ('bool_mask','false'), ('float_mask','false'), ('has_sinks','false')]:
    source = re.sub(r'constant bool ' + name + r' \[\[function_constant\(\d+\)\]\];', 'constant bool ' + name + ' = ' + value + ';', source)
source = re.sub(r'constant int blocks \[\[function_constant\(26\)\]\];', 'constant int blocks = 64;', source)
source = re.sub(r', function_constant\(\w+\)', '', source)
source = source.replace('constant bool has_mask = false;\n', '')
# Keep the arithmetic body verbatim while reducing this model's host ABI to
# consecutive buffers followed by one fixed-shape params struct.
source = 'struct AttentionDecodeParams { uint length; uint capacity; };\n' + source
common = """
  const int N = int(p.length);
  const size_t k_head_stride = size_t(p.capacity) * 256;
  const size_t k_seq_stride = 256;
  const size_t v_head_stride = size_t(p.capacity) * 256;
  const size_t v_seq_stride = 256;
  const float scale = 0.0625f;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  const device T* sinks = nullptr;
  const int mask_kv_seq_stride = 0;
  const int mask_q_seq_stride = 0;
  const int mask_head_stride = 0;
"""
geometry = """
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {"""
def replace_signature(name, signature, locals_):
    global source
    start = source.index('[[kernel]] void ' + name + '(')
    end = source.index(') {', start) + 3
    source = source[:start] + '[[kernel]] void ' + name + '(' + signature + locals_ + source[end:]
replace_signature('sdpa_vector', """
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],""" + geometry,
    common + '\n  const int gqa_factor = 4;\n  const int num_q_heads = 8;\n')
replace_signature('sdpa_vector_2pass_1', """
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant AttentionDecodeParams& p [[buffer(6)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {""", common)
replace_signature('sdpa_vector_2pass_2', """
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],""" + geometry, '\n  (void)p;\n')
# Clone the single-pass route for <=8 causal queries and token-major output.
small_start = source.index('template <typename T, int D, int V = D>')
small_end = source.index('template <typename T, int D, int V = D, int TK = 4>')
small = source[small_start:small_end].replace('void sdpa_vector(', 'void sdpa_vector_small(')
small = small.replace('if (do_causal)', 'if (true)')
small = small.replace('const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;', 'const int o_offset = q_seq_idx * tpg.x + q_batch_head_idx;')
source += small
source += '''\ntemplate [[host_name("attn_small")]] [[kernel]]
decltype(sdpa_vector_small<bfloat, 256>) sdpa_vector_small<bfloat, 256>;
template [[host_name("attn_decode")]] [[kernel]]
decltype(sdpa_vector<bfloat, 256>) sdpa_vector<bfloat, 256>;
template [[host_name("attn_decode_split1")]] [[kernel]]
decltype(sdpa_vector_2pass_1<bfloat, 256>) sdpa_vector_2pass_1<bfloat, 256>;
template [[host_name("attn_decode_split2")]] [[kernel]]
decltype(sdpa_vector_2pass_2<bfloat, 256>) sdpa_vector_2pass_2<bfloat, 256>;
'''
(output / 'attention_decode.metal').write_text(license_comment + prefix + source)
