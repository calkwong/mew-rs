module types;

public struct ObjectData {
  public float4x4 world_transform;
  public uint mesh_id;
}

public struct Mesh {
  public uint vertex_offset;
  public uint first_index;
  public uint index_count;
  public float padding;
}

public struct Vertex {
  public float3 pos;
  float padding;
  public float3 normal;
  float padding2;
}

public struct DrawIndirect {
  public uint index_count;
  public uint instance_count;
  public uint first_index;
  public int vertex_offset;
  public uint first_instance;
  uint padding[3];
}
