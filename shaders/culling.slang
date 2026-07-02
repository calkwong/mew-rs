import types;

struct PushConstants {
  float4x4 view;
  Mesh *mesh_buffer;
  ObjectData *object_buffer;
  DrawIndirect *draw_indirect_buffer;
  Dispatch *dispatch_buffer;
  float4 planes;
  float p00;
  float p11;
  float near;
  float far;
  uint count;
}

[[vk::push_constant]]
PushConstants pc;

// Symmetrical sphere-plane distance test
bool frustum_culled(float3 center, float radius) {
  bool visible = true;

  visible = visible &&
            (center.z * pc.planes.y - abs(center.x) * pc.planes.x) >= -radius;
  visible = visible &&
            (center.z * pc.planes.w - abs(center.y) * pc.planes.z) >= -radius;
  visible = visible && (center.z - radius <= -pc.near) &&
            (center.z + radius >= -pc.far);

  return !visible;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID) {
  if (global_id.x >= pc.count)
    return;

  uint index = global_id.x;

  ObjectData o = pc.object_buffer[index];
  uint mesh_id = o.mesh_id;

  Mesh mesh = pc.mesh_buffer[mesh_id];

  float4 center = mul(o.world_transform, float4(mesh.center, 1));
  center = mul(pc.view, center);

  float3x3 transform = float3x3(o.world_transform);
  float scale = max(max(length(transform[0]), length(transform[1])),
                    length(transform[2]));

  float radius = scale * mesh.radius;

  if (!frustum_culled(center.xyz, radius)) {
    uint count;
    InterlockedAdd(pc.dispatch_buffer.x, 1, count);
    pc.dispatch_buffer.y = 1;
    pc.dispatch_buffer.z = 1;

    pc.draw_indirect_buffer[count].index_count = mesh.index_count;
    pc.draw_indirect_buffer[count].instance_count = 1;
    pc.draw_indirect_buffer[count].first_index = mesh.first_index;
    pc.draw_indirect_buffer[count].vertex_offset = mesh.vertex_offset;
    pc.draw_indirect_buffer[count].first_instance = index;
  }
}
