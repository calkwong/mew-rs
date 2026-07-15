use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;

/// Descriptor handles
/// 6 | 2 | 3 | 21
/// Version, DepthFlag, ImportFlag, RenderResourceTag, DescriptorHandle

// Must be representable with 3 bits, otherwise amend how descriptor handles are constructed
#[derive(Clone, Copy)]
pub enum RenderResourceTag {
    UniformBuffer = 0, // This is tied to binding slot of buffer descriptor set, DO NOT CHANGE!
    StorageBuffer = 1, // This is tied to binding slot of buffer descriptor set, DO NOT CHANGE!
    StorageImage = 2,
    SampledImage = 3,
    Sampler = 4,
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

pub type ResourceTracker = Arc<Mutex<CountTracker>>;

pub struct DescriptorHandle(pub u32);

impl DescriptorHandle {
    const HANDLE_MASK: u32 = (1u32 << 21) - 1;
    const TAG_MASK: u32 = ((1u32 << 3) - 1) << 21;

    pub fn handle(&self) -> u32 {
        self.0 & Self::HANDLE_MASK
    }

    pub fn tag(&self) -> u32 {
        (self.0 & Self::TAG_MASK) >> 21
    }

    pub fn set_tag(&mut self, tag: RenderResourceTag) {
        self.0 |= (tag as u32) << 21;
    }
}

pub struct Descriptor {
    pub pool: vk::DescriptorPool,
    pub sets: [vk::DescriptorSet; 4],
    pub layouts: [vk::DescriptorSetLayout; 4],

    pub ubo_handle: ResourceTracker,
    pub ssbo_handle: ResourceTracker,
    pub image_handle: ResourceTracker,
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
            ubo_handle: Arc::new(Mutex::new(CountTracker { next: 0 })),
            ssbo_handle: Arc::new(Mutex::new(CountTracker { next: 0 })),
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

// TODO: possibly move into backend?
pub fn allocate_descriptor_handle(
    descriptor: &Descriptor,
    tag: RenderResourceTag,
) -> DescriptorHandle {
    let mut handle: DescriptorHandle = match tag {
        RenderResourceTag::UniformBuffer => {
            DescriptorHandle(descriptor.ubo_handle.lock().unwrap().add())
        }
        RenderResourceTag::StorageBuffer => {
            DescriptorHandle(descriptor.ssbo_handle.lock().unwrap().add())
        }
        RenderResourceTag::SampledImage => {
            DescriptorHandle(descriptor.image_handle.lock().unwrap().add())
        }
        RenderResourceTag::StorageImage => {
            DescriptorHandle(descriptor.image_handle.lock().unwrap().add())
        }
        _ => panic!("Not implemented"),
    };

    // Set handle tag
    handle.set_tag(tag as _);
    handle
}

pub fn get_descriptor_index(tag: RenderResourceTag) -> usize {
    match tag {
        RenderResourceTag::UniformBuffer => 0,
        RenderResourceTag::StorageBuffer => 0,
        RenderResourceTag::StorageImage => 1,
        RenderResourceTag::SampledImage => 2,
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
    binding: &[vk::DescriptorSetLayoutBinding],
    binding_flags: &[vk::DescriptorBindingFlags],
) -> vk::DescriptorSetLayout {
    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .push_next(&mut binding_flags_info)
        .bindings(binding);

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
