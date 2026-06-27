[vk::binding(0, 0)]
RWTexture2D draw_image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 t_id : SV_DispatchThreadID
) {

    uint2 draw_image_dims;
    draw_image.GetDimensions(draw_image_dims.x, draw_image_dims.y);

    if (any(t_id.xy >= draw_image_dims))
        return;

    float2 uv = float2(t_id.xy) / draw_image_dims;

    draw_image.Store(t_id.xy, float4(uv, 0.0, 1.0));
}
