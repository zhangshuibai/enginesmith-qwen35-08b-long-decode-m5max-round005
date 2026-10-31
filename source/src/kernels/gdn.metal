// GatedDeltaNet recurrence adapted from mlx-lm 0.32.2 gated_delta.py.
// Copyright (c) 2025 Apple Inc. MIT license; see GDN_INTERFACE.md.
#include <metal_stdlib>
using namespace metal;

struct GdnParams { uint rows; };

// MLX's BF16 elementary operations round after each operation, including the
// exp and denominator inside sigmoid. These boundaries matter for this model.
inline float gdn_sigmoid_bf(float x) {
    float e = float(bfloat(precise::exp(abs(x))));
    float d = float(bfloat(1.0f + e));
    float y = float(bfloat(1.0f / d));
    return float(bfloat(x < 0.0f ? y : 1.0f - y));
}

// MLX utils.h log1p, specialized to x in [0,1] (the softplus exponential).
// Metal's standard library does not provide log1p.
inline float gdn_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == 1.0f) return x;
    return x * (log(xp1) / (xp1 - 1.0f));
}

inline float gdn_softplus_bf(float x) {
    float e = float(bfloat(exp(-abs(x))));
    float l = float(bfloat(gdn_log1p(e)));
    return float(bfloat(max(x, 0.0f) + l));
}

// qkv: [T,6144], history: [3,6144], weight: [6144,4,1].
// Convolution and SiLU have distinct BF16 boundaries.
kernel void gdn_conv(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *history [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *conv [[buffer(3)]],
    constant GdnParams &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * 6144u) return;
    uint t = index / 6144u;
    uint c = index % 6144u;
    float acc = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        int row = int(t) + int(tap) - 3;
        float x = row < 0 ? float(history[uint(row + 3) * 6144u + c])
                          : float(qkv[uint(row) * 6144u + c]);
        acc = fma(x, float(weight[c * 4u + tap]), acc);
    }
    float x = float(bfloat(acc));
    conv[index] = bfloat(x * gdn_sigmoid_bf(x));
}

// Run only after gdn_conv completes. Each thread owns all history for its
// channel, so T=1 and T=2 updates cannot race with one another.
kernel void gdn_history(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *history [[buffer(1)]],
    constant GdnParams &p [[buffer(2)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= 6144u || p.rows == 0) return;
    bfloat h0 = history[c];
    bfloat h1 = history[6144u + c];
    bfloat h2 = history[12288u + c];
    bfloat old[3] = { h0, h1, h2 };
    for (uint j = 0; j < 3; ++j) {
        int row = int(p.rows) + int(j) - 3;
        history[j * 6144u + c] = row < 0 ? old[row + 3]
            : qkv[uint(row) * 6144u + c];
    }
}

// One SIMD-group for each [token,head]. Normalize Q and K in place, leave V
// unchanged, and produce scalar recurrence gates.
kernel void gdn_prepare(
    device bfloat *qkv [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const float *a_log [[buffer(3)]],
    device const bfloat *dt_bias [[buffer(4)]],
    device float *g [[buffer(5)]],
    device bfloat *beta [[buffer(6)]],
    constant GdnParams &p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (group >= p.rows * 16u) return;
    uint t = group / 16u;
    uint h = group % 16u;
    uint off = t * 6144u + h * 128u + lane * 4u;
    float q[4], k[4];
    float qs = 0.0f, ks = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        q[i] = float(qkv[off + i]);
        k[i] = float(qkv[off + 2048u + i]);
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    qs = simd_sum(qs);
    ks = simd_sum(ks);
    float qi = precise::rsqrt(qs / 128.0f + 1.0e-6f / 128.0f);
    float ki = precise::rsqrt(ks / 128.0f + 1.0e-6f / 128.0f);
    // Scalars multiplied into BF16 arrays are themselves BF16 in MLX.
    const float q_scale = 0.0078125f;
    const float k_scale = float(bfloat(0.08838834764831845f));
    for (uint i = 0; i < 4; ++i) {
        qkv[off + i] = bfloat(float(bfloat(q[i] * qi)) * q_scale);
        qkv[off + 2048u + i] = bfloat(float(bfloat(k[i] * ki)) * k_scale);
    }
    if (lane == 0) {
        float ax = float(bfloat(float(a[group]) + float(dt_bias[h])));
        g[group] = precise::exp(-precise::exp(a_log[h]) * gdn_softplus_bf(ax));
        beta[group] = bfloat(gdn_sigmoid_bf(float(b[group])));
    }
}

// Single-token equivalent of gdn_conv, gdn_history, then gdn_prepare.
// One SIMD group owns a [Q/K/V block, head], with the same four adjacent
// elements per lane and SIMD reduction order as gdn_prepare. Each channel's
// convolution history is read and shifted by its sole owning lane, so there
// is no cross-threadgroup read/write dependency. Dispatch 48 groups x 32.
kernel void gdn_conv_prepare_decode(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *history [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *conv [[buffer(3)]],
    device const bfloat *a [[buffer(4)]],
    device const bfloat *b [[buffer(5)]],
    device const float *a_log [[buffer(6)]],
    device const bfloat *dt_bias [[buffer(7)]],
    device float *g [[buffer(8)]],
    device bfloat *beta [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint block = group / 16u;
    uint h = group % 16u;
    uint off = block * 2048u + h * 128u + lane * 4u;
    float v[4];
    float ss = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint c = off + i;
        bfloat old[3] = { history[c], history[6144u + c], history[12288u + c] };
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            float x = tap < 3 ? float(old[tap]) : float(qkv[c]);
            acc = fma(x, float(weight[c * 4u + tap]), acc);
        }
        float x = float(bfloat(acc));
        v[i] = float(bfloat(x * gdn_sigmoid_bf(x)));
        ss += v[i] * v[i];
        history[c] = old[1];
        history[6144u + c] = old[2];
        history[12288u + c] = qkv[c];
    }
    if (block < 2u) {
        ss = simd_sum(ss);
        float inv = precise::rsqrt(ss / 128.0f + 1.0e-6f / 128.0f);
        const float q_scale = 0.0078125f;
        const float k_scale = float(bfloat(0.08838834764831845f));
        float scale = block == 0u ? q_scale : k_scale;
        for (uint i = 0; i < 4; ++i) {
            conv[off + i] = bfloat(float(bfloat(v[i] * inv)) * scale);
        }
    } else {
        for (uint i = 0; i < 4; ++i) conv[off + i] = bfloat(v[i]);
    }
    if (block == 0u && lane == 0u) {
        float ax = float(bfloat(float(a[h]) + float(dt_bias[h])));
        g[h] = precise::exp(-precise::exp(a_log[h]) * gdn_softplus_bf(ax));
        beta[h] = bfloat(gdn_sigmoid_bf(float(b[h])));
    }
}

// Port of MLX's packed explicit-tree kernel. Four lanes own a value row;
// eight value rows occupy a SIMD-group. Each lane retains 32 FP32 state
// elements across every timestep. The state buffer can be updated in place.
kernel void gdn_recur(
    device const bfloat *qkv [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const bfloat *beta [[buffer(2)]],
    device float *state_io [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant GdnParams &p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint h = gid.z;
    uint row = gid.y * 8u + lane / 4u;
    uint lane_in_row = lane & 3u;
    uint key_offset = lane_in_row * 32u;
    uint state_offset = (h * 128u + row) * 128u + key_offset;
    float state[32];
    for (int i = 0; i < 32; ++i) state[i] = state_io[state_offset + i];
    for (uint t = 0; t < p.rows; ++t) {
        uint q_offset = t * 6144u + h * 128u + key_offset;
        uint k_offset = q_offset + 2048u;
        uint v_offset = t * 6144u + 4096u + h * 128u + row;
        float gt = g[t * 16u + h];
        float part[8];
        for (int pb = 0; pb < 8; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] * gt;
                acc += state[e] * float(qkv[k_offset + e]);
            }
            part[pb] = acc;
        }
        float memory = ((part[0] + part[1]) + (part[2] + part[3])) +
                       ((part[4] + part[5]) + (part[6] + part[7]));
        memory += simd_shuffle_xor(memory, 1);
        memory += simd_shuffle_xor(memory, 2);
        float delta = (float(qkv[v_offset]) - memory) * float(beta[t * 16u + h]);
        for (int pb = 0; pb < 8; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] + float(qkv[k_offset + e]) * delta;
                acc += state[e] * float(qkv[q_offset + e]);
            }
            part[pb] = acc;
        }
        float out = ((part[0] + part[1]) + (part[2] + part[3])) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (lane_in_row == 0) y[t * 2048u + h * 128u + row] = bfloat(out);
    }
    for (int i = 0; i < 32; ++i) state_io[state_offset + i] = state[i];
}

kernel void gdn_norm_gate(
    device const bfloat *y [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant GdnParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (group >= p.rows * 16u) return;
    uint off = group * 128u + lane * 4u;
    float v[4];
    float ss = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        v[i] = float(y[off + i]);
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    float inv = precise::rsqrt(ss / 128.0f + 1.0e-6f);
    for (uint i = 0; i < 4; ++i) {
        float norm = float(bfloat(float(bfloat(v[i] * inv)) * float(weight[lane * 4u + i])));
        float gate = float(z[off + i]);
        float sy = 1.0f / (1.0f + precise::exp(abs(gate)));
        float sig = gate < 0.0f ? sy : 1.0f - sy;
        out[off + i] = bfloat((gate * sig) * norm);
    }
}
