module types;

public struct ObjectData
{
    public float4x4 world_transform;
    public uint mesh_id;
    public uint material_id;
    uint padding[2];
}

public struct Mesh
{
    public uint vertex_offset;
    public uint first_index;
    public uint index_count;
    public float radius;
    public float3 center;
}

public struct Vertex
{
    public float3 pos;
    public float uv_x;
    public float3 normal;
    public float uv_y;
}

public struct DrawIndirect
{
    public uint index_count;
    public uint instance_count;
    public uint first_index;
    public int vertex_offset;
    public uint first_instance;
}

public struct Dispatch
{
    public uint x;
    public uint y;
    public uint z;
}

// TODO: investigate scalar layout issue
public struct MaterialData {
    public float4 base_color_factor;
    public uint diffuse_id;
    uint padding[3];
}
