import types;

// TODO: move to a module
[vk::binding(0, 0)]
__DynamicResource<__DynamicResourceKind.General> storage_images[];

public T get_storage_image<T>(uint handle)
    where T : IOpaqueDescriptor {
  return storage_images[handle].asOpaqueDescriptor<T>();
}

struct Interp {
  float4 position : SV_Position;
  float3 normal;
}

struct PushConstants {
  Vertex *vertex_buffer;
  Mesh *mesh_buffer;
  ObjectData *object_buffer;
}

[[vk::push_constant]]
PushConstants pc;

// TODO: some system values may req Vulkan ver
[shader("vertex")]
Interp vs_main(uint vertex_id: SV_VertexID,
               uint base_vertex: SV_StartVertexLocation) {
  Interp interp;

  Vertex vertex = pc.vertex_buffer[vertex_id];
  // TODO: mvp
  interp.position = mul(float4x4(1.0), float4(vertex.pos, 1.0));
  interp.normal = vertex.normal;

  return interp;
}

[shader("pixel")]
void ps_main(Interp in, out float4 out_color: SV_Target0) {
  out_color = float4(in.normal, 1.0);
  out_color.xyz = out_color.xyz * 0.5 + 0.5;
}
