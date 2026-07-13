import bindings;

struct PushConstants
{
    uint src_id;
    uint dst_id;
}

[[vk::push_constant]]
PushConstants pc;

float3 linear_to_srgb(float3 c)
{
	return pow(c, float3(1.0 / 2.2));
}

float3 reinhard(float3 c)
{
	return c / (float3(1.0) + c);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 t_id: SV_DispatchThreadID)
{
    RWTexture2D src_image = get_storage_image<RWTexture2D>(pc.src_id);
    RWTexture2D dst_image = get_storage_image<RWTexture2D>(pc.dst_id);
    uint2 src_dims;
    src_image.GetDimensions(src_dims.x, src_dims.y);

    if (any(t_id.xy >= src_dims))
        return;

    float4 color = src_image.Load(t_id.xy);

    // color.xyz = linear_to_srgb(reinhard(color.xyz));

    dst_image.Store(t_id.xy, float4(color.xyz, 1.0));
}
