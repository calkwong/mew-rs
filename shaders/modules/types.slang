module types;

public struct ObjectData {
    public float4x4 world_transform;
}

public struct Mesh {
    public uint vertex_offset;
    public uint first_index;
    public uint index_count;
}

public struct Vertex {
    public float3 pos;
    public float padding;
    public float3 normal;
}
