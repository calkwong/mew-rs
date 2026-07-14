// Lightly modified version of https://github.com/GPUOpen-Effects/FidelityFX-SPD/blob/master/ffx-spd/ffx_spd.h
//
// Limitations:
// - Strictly PoT only
// - No texture array/cubemap support
// - Base mip must be next PoT because of the way SPD works, strictly requiring a 2 to 1 mapping (handwavy explanation)
// - Modified for hiz, will need adjustment for general use

import bindings;

// TODO: globallycoherent variant of get_storage_image
[[vk::binding(0, 1)]][vk::image_format("r32f")] globallycoherent RWTexture2D<float> images_r32f_coherent[];

struct PushConstants
{
    uint* spd_counter_buffer;
	float2 rcp_resolution; // resolution of src image mapped to next PoT
    uint mips;
	uint num_wgs;
	uint src_id; // depth texture to sample
	uint dst_id; // mip level 0 of hi-z we write to, offset accordingly!
};

[[vk::push_constant]] PushConstants pc;

groupshared float shared_intermediate[16][16];
groupshared uint spd_counter;

uint ABfe(uint src, uint off, uint bits)
{
    return bitfieldExtract(src, off, bits);
}

uint ABfiM(uint src, uint ins, uint bits)
{
    return bitfieldInsert(src, ins, 0, bits);
}

//  LANE TO 8x8 MAPPING
//  ===================
//  00 01 08 09 10 11 18 19
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f
//  20 21 28 29 30 31 38 39
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f
uint2 ARmpRed8x8(uint a)
{
    return uint2(ABfiM(ABfe(a, 2u, 3u), a, 1u), ABfiM(ABfe(a, 3u, 3u), ABfe(a, 1u, 2u), 2u));
}

float sample_texture(Texture2D<float> tex, SamplerState samp, float2 uv, uint slice)
{
    return tex.SampleLevel(samp, uv, 0.0).r;
}

void spd_store_coherent(int2 coords, float value, uint slice)
{
    images_r32f_coherent[pc.dst_id + 5].Store(coords, value);
}

void spd_store(int2 coords, float value, uint mip, uint slice)
{
    RWTexture2D<float> tex = get_storage_image<RWTexture2D<float>>(pc.dst_id + mip);
    tex.Store(coords, value);
}

void spd_store_intermediate(uint x, uint y, float value)
{
    shared_intermediate[x][y] = value;
}

// switch to this when we want min and max depth
float spd_load(int2 coords, uint mip, uint slice)
{
    RWTexture2D<float> tex = get_storage_image<RWTexture2D<float>>(pc.dst_id + mip);
    return tex.Load(coords).r;
}

float spd_load_intermediate(uint x, uint y)
{
    return shared_intermediate[x][y];
}

float spd_reduce_4(float v0, float v1, float v2, float v3)
{
    return min(v0, min(v1, min(v2, v3)));
}

float spd_reduce_quad(float v)
{
    float v0 = v;
    float v1 = QuadReadAcrossX(v);
    float v2 = QuadReadAcrossY(v);
    float v3 = QuadReadAcrossDiagonal(v);

    return spd_reduce_4(v0, v1, v2, v3);
}

float spd_reduce_coherent_load(int2 base, uint slice)
{
    uint idx = pc.dst_id + 5;

    float v0 = images_r32f_coherent[idx].Load(base).r;
    float v1 = images_r32f_coherent[idx].Load(base + int2(0, 1)).r;
    float v2 = images_r32f_coherent[idx].Load(base + int2(1, 0)).r;
    float v3 = images_r32f_coherent[idx].Load(base + int2(1, 1)).r;

    return spd_reduce_4(v0, v1, v2, v3);
}

void spd_downsample_mips_0_1(uint x, uint y, uint2 wg_id, uint local_id, uint mips, uint slice)
{
    SamplerState samp = get_sampler<SamplerState>(DEPTH_REDUCE);
    Texture2D<float> depth_tex = get_sampled_image<Texture2D<float>>(pc.src_id);

    float v[4];

    int2 tex = int2(wg_id * 64) + int2(x * 2, y * 2);
    float2 uv = (float2(tex) + float2(0.5)) * pc.rcp_resolution;
    int2 pix = int2(wg_id * 32) + int2(x, y);
    v[0] = sample_texture(depth_tex, samp, uv, 0);
    spd_store(pix, v[0], 0, slice);

    tex = int2(wg_id * 64) + int2(x * 2 + 32, y * 2);
    uv = (float2(tex) + float2(0.5)) * pc.rcp_resolution;
    pix = int2(wg_id * 32) + int2(x + 16, y);
    v[1] = sample_texture(depth_tex, samp, uv, 0);
    spd_store(pix, v[1], 0, slice);

    tex = int2(wg_id * 64) + int2(x * 2, y * 2 + 32);
    uv = (float2(tex) + float2(0.5)) * pc.rcp_resolution;
    pix = int2(wg_id * 32) + int2(x, y + 16);
    v[2] = sample_texture(depth_tex, samp, uv, 0);
    spd_store(pix, v[2], 0, slice);

    tex = int2(wg_id * 64) + int2(x * 2 + 32, y * 2 + 32);
    uv = (float2(tex) + float2(0.5)) * pc.rcp_resolution;
    pix = int2(wg_id * 32) + int2(x + 16, y + 16);
    v[3] = sample_texture(depth_tex, samp, uv, 0);
    spd_store(pix, v[3], 0, slice);

    if (mips <= 1)
        return;

    v[0] = spd_reduce_quad(v[0]);
    v[1] = spd_reduce_quad(v[1]);
    v[2] = spd_reduce_quad(v[2]);
    v[3] = spd_reduce_quad(v[3]);

    if ((local_id % 4) == 0)
    {
        pix = int2(wg_id) * 16 + int2(x / 2, y / 2);
        spd_store(pix, v[0], 1, slice);
        spd_store_intermediate(x / 2, y / 2, v[0]);

        pix = int2(wg_id) * 16 + int2(x / 2 + 8, y / 2);
        spd_store(pix, v[1], 1, slice);
        spd_store_intermediate(x / 2 + 8, y / 2, v[1]);

        pix = int2(wg_id) * 16 + int2(x / 2, y / 2 + 8);
        spd_store(pix, v[2], 1, slice);
        spd_store_intermediate(x / 2, y / 2 + 8, v[2]);

        pix = int2(wg_id) * 16 + int2(x / 2 + 8, y / 2 + 8);
        spd_store(pix, v[3], 1, slice);
        spd_store_intermediate(x / 2 + 8, y / 2 + 8, v[3]);
    }
}

void spd_downsample_mip_2(uint x, uint y, uint2 wg_id, uint local_id, uint mip, uint slice)
{
    float v = spd_load_intermediate(x, y);
    v = spd_reduce_quad(v);

    if (local_id % 4 == 0)
    {
        spd_store(int2(wg_id * 8) + int2(x / 2, y / 2), v, mip, slice);
        spd_store_intermediate(x + (y / 2) % 2, y, v);
    }
}

void spd_downsample_mip_3(uint x, uint y, uint2 wg_id, uint local_id, uint mip, uint slice)
{
    if (local_id < 64)
    {
        float v = spd_load_intermediate(x * 2 + y % 2, y * 2);
        v = spd_reduce_quad(v);

        if (local_id % 4 == 0)
        {
            spd_store(int2(wg_id * 4) + int2(x / 2, y / 2), v, mip, slice);
            spd_store_intermediate(x * 2 + y / 2, y * 2, v);
        }
    }
}

void spd_downsample_mip_4(uint x, uint y, uint2 wg_id, uint local_id, uint mip, uint slice)
{
    if (local_id < 16)
    {
        float v = spd_load_intermediate(x * 4 + y, y * 4);
        v = spd_reduce_quad(v);

        if (local_id % 4 == 0)
        {
            spd_store(int2(wg_id * 2) + int2(x / 2, y / 2), v, mip, slice);
            spd_store_intermediate(x / 2 + y, 0, v);
        }
    }
}

void spd_downsample_mip_5(uint2 wg_id, uint local_id, uint mip, uint slice)
{
    if (local_id < 4)
    {
        float v = spd_load_intermediate(local_id, 0);
        v = spd_reduce_quad(v);

        if (local_id % 4 == 0)
        {
            if (mip == 5)
                spd_store_coherent(int2(wg_id), v, slice);
            else
                spd_store(int2(wg_id), v, mip, slice);
        }
    }
}

void spd_downsample_next_four(uint x, uint y, uint2 wg_id, uint local_id, uint base_mip, uint mips, uint slice)
{
    if (mips <= base_mip) return;
    GroupMemoryBarrierWithGroupSync();
    spd_downsample_mip_2(x, y, wg_id, local_id, base_mip, slice);

    if (mips <= base_mip + 1) return;
    GroupMemoryBarrierWithGroupSync();
    spd_downsample_mip_3(x, y, wg_id, local_id, base_mip + 1, slice);

    if (mips <= base_mip + 2) return;
    GroupMemoryBarrierWithGroupSync();
    spd_downsample_mip_4(x, y, wg_id, local_id, base_mip + 2, slice);

    if (mips <= base_mip + 3) return;
    GroupMemoryBarrierWithGroupSync();
    spd_downsample_mip_5(wg_id, local_id, base_mip + 3, slice);
}

bool spd_exit_workgroup(uint num_wgs, uint local_id, uint slice)
{
    // spd_counter must be shared, not local
    if (local_id == 0)
        InterlockedAdd(*pc.spd_counter_buffer, 1, spd_counter);

    GroupMemoryBarrierWithGroupSync();

    // only 1 wg to proceed
    return spd_counter != (num_wgs - 1);
}

void spd_reset_atomic_counter(uint slice)
{
    *pc.spd_counter_buffer = 0;
}

void spd_downsample_mips_6_7(uint x, uint y, uint mips, uint slice)
{
    int2 tex = int2(x * 4 + 0, y * 4 + 0);
    int2 pix = int2(x * 2 + 0, y * 2 + 0);
    float v0 = spd_reduce_coherent_load(tex, slice);
    spd_store(pix, v0, 6, slice);

    tex = int2(x * 4 + 2, y * 4 + 0);
    pix = int2(x * 2 + 1, y * 2 + 0);
    float v1 = spd_reduce_coherent_load(tex, slice);
    spd_store(pix, v1, 6, slice);

    tex = int2(x * 4 + 0, y * 4 + 2);
    pix = int2(x * 2 + 0, y * 2 + 1);
    float v2 = spd_reduce_coherent_load(tex, slice);
    spd_store(pix, v2, 6, slice);

    tex = int2(x * 4 + 2, y * 4 + 2);
    pix = int2(x * 2 + 1, y * 2 + 1);
    float v3 = spd_reduce_coherent_load(tex, slice);
    spd_store(pix, v3, 6, slice);

    if (mips <= 7) return;
    // no barrier needed, working on values only from the same thread

    float v = spd_reduce_4(v0, v1, v2, v3);
    spd_store(int2(x, y), v, 7, slice);
    spd_store_intermediate(x, y, v);
}

void spd_downsample(uint2 wg_id, uint local_id, uint mips, uint num_wgs, uint slice)
{
    // map to 2d shared memory
    uint2 sub_xy = ARmpRed8x8(local_id % 64);
    uint x = sub_xy.x + 8 * ((local_id >> 6) % 2);
    uint y = sub_xy.y + 8 * ((local_id >> 7));

    spd_downsample_mips_0_1(x, y, wg_id, local_id, mips, slice);

    spd_downsample_next_four(x, y, wg_id, local_id, 2, mips, slice);

    if (mips <= 6) return;

    if (spd_exit_workgroup(num_wgs, local_id, slice)) return;

    // only 1 wg from here on out

    if (local_id == 0)
        spd_reset_atomic_counter(slice);

    spd_downsample_mips_6_7(x, y, mips, slice);
    spd_downsample_next_four(x, y, uint2(0, 0), local_id, 8, mips, slice);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 wg_id : SV_GroupID,
    uint t_index : SV_GroupIndex
)
{
    spd_downsample(
        wg_id.xy,
        t_index,
        pc.mips,
        pc.num_wgs,
        wg_id.z
    );
}
