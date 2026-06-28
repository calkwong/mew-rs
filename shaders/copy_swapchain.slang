// TODO: move to a module
[vk::binding(0, 0)]
__DynamicResource<__DynamicResourceKind.General> storage_images[];

public T get_storage_image<T>(uint handle)
    where T : IOpaqueDescriptor {
  return storage_images[handle].asOpaqueDescriptor<T>();
}

struct PushConstants {
  uint src_id;
  uint dst_id;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 t_id: SV_DispatchThreadID) {

  RWTexture2D src_image = get_storage_image<RWTexture2D>(pc.src_id);
  RWTexture2D dst_image = get_storage_image<RWTexture2D>(pc.dst_id);
  uint2 src_dims;
  src_image.GetDimensions(src_dims.x, src_dims.y);

  if (any(t_id.xy >= src_dims))
    return;

  float4 color = src_image.Load(t_id.xy);

  dst_image.Store(t_id.xy, color);
}
