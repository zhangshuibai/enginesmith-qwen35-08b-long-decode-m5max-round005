/*
Q/K normalization, rotary arithmetic and sigmoid adapted from MLX 0.32.4.
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#include <metal_stdlib>
using namespace metal;

// All input/output matrices are row-major BF16. Cache is [2,capacity,256].
struct AttentionPrepareParams {
    uint tokens;
    uint offset;
    uint capacity;
};

// Q/K RMSNorm keeps the reference's BF16 normalized-value boundary before gain.
inline float attn_head_inv_rms(
    const device bfloat* x,
    uint thread_id,
    uint lane,
    uint simd_id,
    threadgroup float* sums)
{
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(x[4 * thread_id + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) sums[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return precise::rsqrt((sums[0] + sums[1]) / 256.0f + 1e-6f);
}

inline bfloat attn_rope_component(
    threadgroup const bfloat* normalized,
    uint d,
    uint position)
{
    if (d >= 64) return normalized[d];
    uint pair = d % 32;
    // std::log2(10000000.0f), rounded once to float by the MLX launcher.
    float inv_freq = metal::exp2(-(float(pair) / 32.0f) * 23.253496170043945f);
    float angle = float(position) * inv_freq;
    float c = metal::fast::cos(angle);
    float s = metal::fast::sin(angle);
    float lo = float(normalized[pair]);
    float hi = float(normalized[pair + 32]);
    return bfloat(d < 32 ? lo * c - hi * s : lo * s + hi * c);
}

kernel void attn_prepare_q(
    const device bfloat* projected [[buffer(0)]],
    const device bfloat* gain [[buffer(1)]],
    device bfloat* queries [[buffer(2)]],
    constant AttentionPrepareParams& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const uint head = group.x;
    const uint token = group.y;
    const device bfloat* x = projected + ulong(token) * 4096 + head * 512;
    device bfloat* y = queries + ulong(token) * 2048 + head * 256;
    threadgroup float sums[2];
    threadgroup bfloat normalized[256];
    float inv = attn_head_inv_rms(x, thread_id, lane, simd_id, sums);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        normalized[d] = bfloat(float(bfloat(float(x[d]) * inv)) * float(gain[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        y[d] = attn_rope_component(normalized, d, p.offset + token);
    }
}

kernel void attn_prepare_kv(
    const device bfloat* projected_k [[buffer(0)]],
    const device bfloat* projected_v [[buffer(1)]],
    const device bfloat* gain [[buffer(2)]],
    device bfloat* keys [[buffer(3)]],
    device bfloat* values [[buffer(4)]],
    constant AttentionPrepareParams& p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const uint head = group.x;
    const uint token = group.y;
    const uint position = p.offset + token;
    const device bfloat* x = projected_k + ulong(token) * 512 + head * 256;
    const device bfloat* v = projected_v + ulong(token) * 512 + head * 256;
    ulong cache_index = (ulong(head) * p.capacity + position) * 256;
    threadgroup float sums[2];
    threadgroup bfloat normalized[256];
    float inv = attn_head_inv_rms(x, thread_id, lane, simd_id, sums);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        normalized[d] = bfloat(float(bfloat(float(x[d]) * inv)) * float(gain[d]));
        values[cache_index + d] = v[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        keys[cache_index + d] = attn_rope_component(normalized, d, position);
    }
}

kernel void attn_gate(
    const device bfloat* attention [[buffer(0)]],
    const device bfloat* projected_q_gate [[buffer(1)]],
    device bfloat* gated [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= tokens * 2048) return;
    uint token = index / 2048;
    uint head = (index % 2048) / 256;
    uint d = index % 256;
    float x = float(projected_q_gate[ulong(token) * 4096 + head * 512 + 256 + d]);
    // MLX unary sigmoid has BF16 boundaries at exp, add, reciprocal, subtract.
    float e = float(bfloat(metal::precise::exp(metal::abs(x))));
    float denom = float(bfloat(1.0f + e));
    float inverse = float(bfloat(1.0f / denom));
    bfloat gate = bfloat(x < 0.0f ? inverse : 1.0f - inverse);
    gated[index] = bfloat(float(attention[index]) * float(gate));
}

// Explicit BF16 score and probability boundaries, matching the unfused
// reference graph used for short full-attention prefills at head dimension256.
// Intended for 9<=tokens<1024 and length<=4096. Scores are [tokens,8,length].
struct AttentionShortParams {
    uint tokens;
    uint length;
    uint capacity;
    uint offset;
};

kernel void attn_short_scores(
    const device bfloat* queries [[buffer(0)]],
    const device bfloat* keys [[buffer(1)]],
    device bfloat* scores [[buffer(2)]],
    constant AttentionShortParams& p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint key_index = group.x * 4 + simd_id;
    uint head = group.y;
    uint token = group.z;
    if (key_index >= p.length) return;
    float acc = -3.3895313892515355e38f;
    if (key_index <= p.offset + token) {
        const device bfloat* q = queries + ulong(token) * 2048 + head * 256 + lane * 8;
        const device bfloat* k = keys + (ulong(head / 4) * p.capacity + key_index) * 256 + lane * 8;
        acc = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            float qi = float(bfloat(float(q[i]) * 0.0625f));
            acc += qi * float(k[i]);
        }
        acc = simd_sum(acc);
    }
    if (lane == 0) scores[(ulong(token) * 8 + head) * p.length + key_index] = bfloat(acc);
}

// MLX softmax_single_row<bfloat,float,4>, specialized to in-place storage and
// model-specific params. It retains the exact reduction and exp arrangement.
kernel void attn_short_softmax(
    device bfloat* scores [[buffer(0)]],
    constant AttentionShortParams& p [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    device bfloat* data = scores + ulong(row) * p.length + thread_id * 4;
    threadgroup float local_max[32];
    threadgroup float local_sum[32];
    float values[4];
    for (uint i = 0; i < 4; ++i) {
        values[i] = thread_id * 4 + i < p.length ? float(data[i]) : -INFINITY;
    }
    if (simd_id == 0) {
        local_max[lane] = -INFINITY;
        local_sum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_value = -3.4028234663852886e38f;
    for (uint i = 0; i < 4; ++i) max_value = max_value < values[i] ? values[i] : max_value;
    max_value = simd_max(max_value);
    if (lane == 0) local_max[simd_id] = max_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        max_value = simd_max(local_max[lane]);
        if (lane == 0) local_max[0] = max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_value = local_max[0];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = metal::fast::exp(values[i] - max_value);
        values[i] = value;
        sum += value;
    }
    sum = simd_sum(sum);
    if (lane == 0) local_sum[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = simd_sum(local_sum[lane]);
        if (lane == 0) local_sum[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = 1.0f / local_sum[0];
    for (uint i = 0; i < 4; ++i) {
        if (thread_id * 4 + i < p.length) data[i] = bfloat(values[i] * inverse);
    }
}

kernel void attn_short_values(
    const device bfloat* scores [[buffer(0)]],
    const device bfloat* values [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant AttentionShortParams& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]])
{
    uint head = group.x;
    uint token = group.y;
    const device bfloat* probabilities = scores + (ulong(token) * 8 + head) * p.length;
    const device bfloat* v = values + ulong(head / 4) * p.capacity * 256 + d;
    float acc = 0.0f;
    for (uint k = 0; k < p.length; ++k) acc += float(probabilities[k]) * float(v[ulong(k) * 256]);
    out[ulong(token) * 2048 + head * 256 + d] = bfloat(acc);
}

// Decode Q and KV preparation share a dispatch with unchanged helpers.
// Q groups 0..7 and KV groups 8..9 write disjoint outputs.
kernel void attn_prepare_qkv_decode(
    const device bfloat* projected [[buffer(0)]],
    const device bfloat* projected_k [[buffer(1)]],
    const device bfloat* projected_v [[buffer(2)]],
    const device bfloat* q_gain [[buffer(3)]],
    const device bfloat* k_gain [[buffer(4)]],
    device bfloat* queries [[buffer(5)]],
    device bfloat* keys [[buffer(6)]],
    device bfloat* values [[buffer(7)]],
    constant AttentionPrepareParams& p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sums[2];
    threadgroup bfloat normalized[256];
    if (group.x < 8) {
        const uint head = group.x;
        const uint token = group.y;
        const device bfloat* x = projected + ulong(token) * 4096 + head * 512;
        device bfloat* y = queries + ulong(token) * 2048 + head * 256;
        float inv = attn_head_inv_rms(x, thread_id, lane, simd_id, sums);
        for (uint i = 0; i < 4; ++i) {
            uint d = 4 * thread_id + i;
            normalized[d] = bfloat(float(bfloat(float(x[d]) * inv)) * float(q_gain[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        y[d] = attn_rope_component(normalized, d, p.offset + token);
    }
    } else {
        const uint head = group.x - 8;
        const uint token = group.y;
        const uint position = p.offset + token;
        const device bfloat* x = projected_k + ulong(token) * 512 + head * 256;
        const device bfloat* v = projected_v + ulong(token) * 512 + head * 256;
        ulong cache_index = (ulong(head) * p.capacity + position) * 256;
        float inv = attn_head_inv_rms(x, thread_id, lane, simd_id, sums);
        for (uint i = 0; i < 4; ++i) {
            uint d = 4 * thread_id + i;
            normalized[d] = bfloat(float(bfloat(float(x[d]) * inv)) * float(k_gain[d]));
            values[cache_index + d] = v[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint d = 4 * thread_id + i;
        keys[cache_index + d] = attn_rope_component(normalized, d, position);
    }
    }
}
