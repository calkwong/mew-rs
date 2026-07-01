import types;

struct PushConstants {
  Mesh *mesh_buffer;
  ObjectData *object_buffer;
  DrawIndirect *draw_indirect_buffer;
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

  DrawIndirect draw;
  draw.index_count = mesh.index_count;
  draw.instance_count = 1;
  draw.first_index = mesh.first_index;
  draw.vertex_offset = mesh.vertex_offset;
  draw.first_instance = index;

  pc.draw_indirect_buffer[index] = draw;
}
