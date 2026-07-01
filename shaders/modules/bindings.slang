[vk::binding(0, 1)]
__DynamicResource<__DynamicResourceKind.General> storage_images[];

[vk::binding(0, 2)]
__DynamicResource<__DynamicResourceKind.General> sampled_images[];

[vk::binding(0, 3)]
__DynamicResource<__DynamicResourceKind.Sampler> samplers[];

public T get_storage_image<T>(uint handle) where T : IOpaqueDescriptor
{
    return storage_images[handle].asOpaqueDescriptor<T>();
}

public T get_sampled_image<T>(uint handle) where T : IOpaqueDescriptor
{
    return sampled_images[handle].asOpaqueDescriptor<T>();
}

public T get_sampler<T>(uint handle) where T : IOpaqueDescriptor
{
    return samplers[handle].asOpaqueDescriptor<T>();
}
