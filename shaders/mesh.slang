import types;

struct Interp
{
    float4 position : SV_Position;
    float3 normal;
}

struct PushConstants
{
    float4x4 view_proj;
    Vertex* vertex_buffer;
    Mesh* mesh_buffer;
    ObjectData* object_buffer;
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

    return interp;
}

[shader("pixel")]
void ps_main(Interp in, out float4 out_color: SV_Target0)
{
    out_color = float4(in.normal, 1.0);
    out_color.xyz = out_color.xyz * 0.5 + 0.5;
}
