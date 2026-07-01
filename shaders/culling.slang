import types;

struct PushConstants {
  Mesh *mesh_buffer;
  ObjectData *object_buffer;
  DrawIndirect* draw_indirect_buffer;
  uint count;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID) {
  if (global_id.x >= pc.count)
    return;

  uint index = global_id.x;

  ObjectData o = pc.object_buffer[index];
  uint mesh_id = o.mesh_id;

  Mesh mesh = pc.mesh_buffer[mesh_id];

  pc.draw_indirect_buffer[index].index_count = mesh.index_count;
  pc.draw_indirect_buffer[index].instance_count = 1;
  pc.draw_indirect_buffer[index].first_index = mesh.first_index;
  pc.draw_indirect_buffer[index].vertex_offset = mesh.vertex_offset;
  pc.draw_indirect_buffer[index].first_instance = index;
}
