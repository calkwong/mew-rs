import types;
import bindings;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
    // float2 uv;
    // nointerpolation uint material_id;
}

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    Mesh* mesh_buffer;
    ObjectData* object_buffer;
    MaterialData* material_buffer;
}

[[vk::push_constant]]
PushConstants pc;

[shader("vertex")]
Interp vs_main(uint instance_id: SV_VulkanInstanceID, uint vertex_id: SV_VulkanVertexID)
{
    Interp interp;

    Vertex vertex = pc.vertex_buffer[vertex_id];
    ObjectData o = pc.object_buffer[instance_id];

    float4 world_pos = mul(o.world_transform, float4(vertex.pos, 1.0));
    interp.position = mul(pc.view_proj, world_pos);
    interp.normal = vertex.normal;
    // interp.uv = float2(vertex.uv_x, vertex.uv_y);
    // interp.material_id = o.material_id;

    return interp;
}

[shader("pixel")]
void ps_main(Interp in, out float4 out_color: SV_Target0)
{
    // MaterialData material = pc.material_buffer[in.material_id];

    // float4 albedo = material.base_color_factor;
    // if (material.diffuse_id != 0)
    // {
    //     Texture2D tex = get_sampled_image<Texture2D>(material.diffuse_id);
    //     SamplerState samp = get_sampler<SamplerState>(0);
    //     albedo *= tex.Sample(samp, in.uv);
    // }

    // out_color = float4(albedo.xyz, 1.0);
    out_color = float4(in.normal, 1.0);
    out_color.xyz = out_color.xyz * 0.5 + 0.5;
}
