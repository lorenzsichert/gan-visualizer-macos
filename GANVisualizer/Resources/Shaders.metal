#include <metal_stdlib>
using namespace metal;

// Must stay in sync with the Swift structs in Renderer.swift / InferenceEngine.swift.

struct InferUniforms {
    uint H;
    uint W;
    int  hue;
    int  doTanh;
};

static inline float mapValue(float v, int doTanh) {
    if (doTanh) { v = tanh(v); }
    return clamp((v + 1.0) * 0.5, 0.0, 1.0);
}

// NCHW float16 planes -> RGBA8 texture, with the RGB hue channel-roll applied.
kernel void planesToRGBA16(device const half *planes [[buffer(0)]],
                           constant InferUniforms &u [[buffer(1)]],
                           texture2d<float, access::write> outTex [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    if (x >= u.W || y >= u.H) return;
    uint plane = u.H * u.W;
    int W = int(u.W), H = int(u.H);
    uint shiftX = uint(((u.hue % W) + W) % W);
    uint shiftY = uint(((u.hue % H) + H) % H);
    uint rx = (x + u.W - shiftX) % u.W;
    uint gy = (y + u.H - shiftY) % u.H;

    float r = mapValue(float(planes[y * u.W + rx]), u.doTanh);
    float g = mapValue(float(planes[plane + gy * u.W + x]), u.doTanh);
    float b = mapValue(float(planes[2 * plane + y * u.W + x]), u.doTanh);
    outTex.write(float4(r, g, b, 1.0), gid);
}

kernel void planesToRGBA32(device const float *planes [[buffer(0)]],
                           constant InferUniforms &u [[buffer(1)]],
                           texture2d<float, access::write> outTex [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    if (x >= u.W || y >= u.H) return;
    uint plane = u.H * u.W;
    int W = int(u.W), H = int(u.H);
    uint shiftX = uint(((u.hue % W) + W) % W);
    uint shiftY = uint(((u.hue % H) + H) % H);
    uint rx = (x + u.W - shiftX) % u.W;
    uint gy = (y + u.H - shiftY) % u.H;

    float r = mapValue(float(planes[y * u.W + rx]), u.doTanh);
    float g = mapValue(float(planes[plane + gy * u.W + x]), u.doTanh);
    float b = mapValue(float(planes[2 * plane + y * u.W + x]), u.doTanh);
    outTex.write(float4(r, g, b, 1.0), gid);
}

// --- Full-window textured quad with reflected mirror tiling -------------------

struct VOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VOut vs_fullscreen(uint vid [[vertex_id]]) {
    float2 p[3] = { float2(-1.0, -3.0), float2(-1.0, 1.0), float2(3.0, 1.0) };
    VOut o;
    float2 pos = p[vid];
    o.pos = float4(pos, 0.0, 1.0);
    o.uv = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return o;
}

struct DrawUniforms {
    float2 viewport;
    float2 imageSize;
    int    mirrorMode;
    float  blurAmount;
};

fragment float4 fs_draw(VOut in [[stage_in]],
                        constant DrawUniforms &u [[buffer(0)]],
                        texture2d<float> tex [[texture(0)]],
                        sampler samp [[sampler(0)]]) {
    const float3 bg = float3(10.0 / 255.0, 11.0 / 255.0, 13.0 / 255.0);

    float2 vp = u.viewport;
    float2 img = u.imageSize;
    float scale = min(vp.x / img.x, vp.y / img.y);
    float2 d = img * scale;
    float2 origin = (vp - d) * 0.5;
    float2 q = (in.uv * vp - origin) / d;

    if (u.mirrorMode == 0) {
        if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return float4(bg, 1.0);
        return float4(tex.sample(samp, q).rgb, 1.0);
    }

    float2 t = floor(q);
    float2 local = q - t;
    if (fmod(t.x + 1024.0, 2.0) >= 1.0) local.x = 1.0 - local.x;
    if (fmod(t.y + 1024.0, 2.0) >= 1.0) local.y = 1.0 - local.y;
    local = clamp(local, 0.0, 1.0);

    bool side = (t.x != 0.0) || (t.y != 0.0);
    if (u.mirrorMode >= 2 && side) {
        float2 texel = 1.0 / img;
        float3 acc = float3(0.0);
        float wsum = 0.0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                float w = exp(-float(i * i + j * j) / 4.0);
                float2 off = float2(float(i), float(j)) * texel * u.blurAmount;
                acc += tex.sample(samp, clamp(local + off, 0.0, 1.0)).rgb * w;
                wsum += w;
            }
        }
        return float4((acc / wsum) * 0.8, 1.0);
    }
    return float4(tex.sample(samp, local).rgb, 1.0);
}
