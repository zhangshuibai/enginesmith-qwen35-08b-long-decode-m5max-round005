// Exact state vector-load/store probe; original recurrence from MLX; see GDN_INTERFACE.md.
#include <metal_stdlib>
using namespace metal;
struct GdnParams { uint rows; };
kernel void gdn_recur_vec005(
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
    for (int i = 0; i < 8; ++i) {
        float4 v = *((device const float4*)(state_io + state_offset) + i);
        state[4*i] = v.x; state[4*i+1] = v.y; state[4*i+2] = v.z; state[4*i+3] = v.w;
    }
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
    for (int i = 0; i < 8; ++i) {
        *((device float4*)(state_io + state_offset) + i) = float4(state[4*i],state[4*i+1],state[4*i+2],state[4*i+3]);
    }
}
