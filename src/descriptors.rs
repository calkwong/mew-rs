use ash::{vk, Device};

pub fn create_descriptor_pool(
    device: &Device,
    pool_size: vk::DescriptorPoolSize,
) -> vk::DescriptorPool {
    let pool_size = [pool_size];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_size);

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    }
}

pub fn create_descriptor_layouts(
    device: &Device,
    binding: vk::DescriptorSetLayoutBinding,
) -> vk::DescriptorSetLayout {
    let binding = [binding];

    let binding_flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
        | vk::DescriptorBindingFlags::PARTIALLY_BOUND];
    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .push_next(&mut binding_flags_info)
        .bindings(&binding);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

pub fn create_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    descriptor_counts: u32,
) -> vk::DescriptorSet {
    let layout = [layout];
    let descriptor_counts = [descriptor_counts];

    let mut variable_alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(&descriptor_counts);

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .push_next(&mut variable_alloc_info)
        .descriptor_pool(pool)
        .set_layouts(&layout);

    unsafe {
        device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap()[0]
    }
}
