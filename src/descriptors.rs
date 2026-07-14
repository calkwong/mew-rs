use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;

pub enum RenderResourceTag {
    Buffer,
    Storage,
    Sampled,
    Sampler,
}

/// Handle recycling not implemented
pub struct CountTracker {
    next: u32,
}

impl CountTracker {
    pub fn add(&mut self) -> u32 {
        let index = self.next;
        self.next += 1;
        index
    }
}

pub type ImageTracker = Arc<Mutex<CountTracker>>;

pub struct Descriptor {
    pub pool: vk::DescriptorPool,
    pub sets: [vk::DescriptorSet; 4],
    pub layouts: [vk::DescriptorSetLayout; 4],

    pub image_handle: ImageTracker,
}

impl Descriptor {
    pub fn new(
        pool: vk::DescriptorPool,
        sets: [vk::DescriptorSet; 4],
        layouts: [vk::DescriptorSetLayout; 4],
    ) -> Self {
        Self {
            pool,
            sets,
            layouts,
            image_handle: Arc::new(Mutex::new(CountTracker { next: 0 })),
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            for layout in self.layouts {
                device.destroy_descriptor_set_layout(layout, None);
            }
        }
    }
}

pub fn get_descriptor_index(tag: RenderResourceTag) -> usize {
    match tag {
        RenderResourceTag::Buffer => 0,
        RenderResourceTag::Storage => 1,
        RenderResourceTag::Sampled => 2,
        RenderResourceTag::Sampler => 3,
    }
}

pub fn create_descriptor_pool(
    device: &ash::Device,
    pool_size: &[vk::DescriptorPoolSize],
) -> vk::DescriptorPool {
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(4)
        .pool_sizes(pool_size);

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    }
}

pub fn create_descriptor_layouts(
    device: &ash::Device,
    binding: vk::DescriptorSetLayoutBinding,
    binding_flags: &[vk::DescriptorBindingFlags],
) -> vk::DescriptorSetLayout {
    let binding = [binding];

    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

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
    device: &ash::Device,
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
