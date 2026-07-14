use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::surface,
    vk::{self, DebugUtilsMessengerEXT, Queue, Semaphore},
};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::*;
use std::{
    borrow::Cow,
    ffi,
    mem::ManuallyDrop,
    os::raw::c_char,
    sync::{Arc, Mutex},
};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

pub mod basic_renderer;
pub mod camera;
pub mod descriptors;
pub mod loader;
pub mod rendergraph;
pub mod swapchain;
use descriptors::{RenderResourceTag, get_descriptor_index};
use swapchain::{Swapchain, create_swapchain};

pub const FRAMES_IN_FLIGHT: usize = 2;
pub const MAX_QUERY_COUNT: u32 = 10;
const MAX_PUSH_CONSTANTS_SIZE: u32 = 256;

#[derive(Default, Copy, Clone)]
pub struct FrameResources {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub image_acquired_semaphore: vk::Semaphore,
    pub pipeline_query: vk::QueryPool,
}

pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub allocation: Allocation,
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub address: vk::DeviceAddress,
    pub allocation: Allocation,
    pub size: vk::DeviceSize,
}

pub struct Device {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub instance: Instance,
    pub graphics_queue: Queue,
    pub properties: vk::PhysicalDeviceProperties,
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    pub frame_resources: [FrameResources; FRAMES_IN_FLIGHT],

    // TODO: put swapchain and render_done_semaphore together?
    pub render_done_semaphores: Vec<Semaphore>,
    pub swapchain: Swapchain,

    // TODO: move to renderer?
    pub descriptor: descriptors::Descriptor,
    pub pipeline_layout: vk::PipelineLayout,

    pub debug_utils_loader: debug_utils::Instance,
    pub debug_callback: DebugUtilsMessengerEXT,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: surface::Instance,
}

impl Device {
    pub fn new(window: &Window) -> Self {
        let entry = Entry::linked();

        // CursorGrabMode::Locked not supported on X11
        // If we fallback to Confined, we get choppy rendering possibly related: https://github.com/rust-windowing/winit/issues/3773
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);

        window.set_cursor_visible(false);

        let window_handle = window
            .window_handle()
            .expect("Window handle error")
            .as_raw();

        let app_name = c"mew-rust";

        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let display_handle = window
            .display_handle()
            .expect("Display handle error")
            .as_raw();

        let mut extension_names = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        extension_names.push(debug_utils::NAME.as_ptr());

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(vk::InstanceCreateFlags::default());

        let instance: Instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Instance creation failed")
        };

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        let debug_callback = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .unwrap()
        };
        let surface_loader = surface::Instance::new(&entry, &instance);
        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };

        let (pdevice, queue_family_index) = unsafe {
            pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();

                            let prop = instance.get_physical_device_properties(*pdevice);
                            let is_discrete_gpu =
                                prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

                            if supports_graphic_and_surface && is_discrete_gpu {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device")
        };

        let queue_family_index = queue_family_index as u32;
        let priorities = [1.0];

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let queue_infos: Vec<vk::DeviceQueueCreateInfo> = vec![queue_create_info];

        // check extension support
        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

        // check feature support
        let mut enabled_vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::default();
        let mut enabled_vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut enabled_vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
        let mut enabled_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enabled_vulkan_1_1_features)
            .push_next(&mut enabled_vulkan_1_2_features)
            .push_next(&mut enabled_vulkan_1_3_features);

        unsafe {
            instance.get_physical_device_features2(pdevice, &mut enabled_features);
        }

        assert!(enabled_features.features.multi_draw_indirect > 0);
        assert!(enabled_vulkan_1_2_features.buffer_device_address > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_partially_bound > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_variable_descriptor_count > 0);
        assert!(enabled_vulkan_1_2_features.draw_indirect_count > 0);
        assert!(enabled_vulkan_1_2_features.scalar_block_layout > 0);
        assert!(enabled_vulkan_1_2_features.runtime_descriptor_array > 0);
        assert!(enabled_vulkan_1_2_features.host_query_reset > 0);
        assert!(enabled_vulkan_1_3_features.synchronization2 > 0);
        assert!(enabled_vulkan_1_3_features.dynamic_rendering > 0);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default()
            .multi_draw_indirect(true)
            .pipeline_statistics_query(true);
        let mut vulkan_1_1_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .draw_indirect_count(true)
            .scalar_block_layout(true)
            .host_query_reset(true);
        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&vulkan_1_0_features)
            .push_next(&mut vulkan_1_1_features)
            .push_next(&mut vulkan_1_2_features)
            .push_next(&mut vulkan_1_3_features);

        let device = unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap()
        };

        let mut properties = vk::PhysicalDeviceProperties2::default();
        unsafe {
            instance.get_physical_device_properties2(pdevice, &mut properties);
        }

        assert_eq!(
            properties.properties.limits.max_push_constants_size,
            MAX_PUSH_CONSTANTS_SIZE
        );

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let swapchain = create_swapchain(
            &instance,
            &surface_loader,
            surface,
            pdevice,
            &device,
            window,
            vk::SwapchainKHR::null(),
        );

        let mut properties = vk::PhysicalDeviceProperties2::default();
        unsafe {
            instance.get_physical_device_properties2(pdevice, &mut properties);
        }

        // Init descriptors
        let pool_size = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 3,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 3,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 300,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 300,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 5,
            },
        ];
        let descriptor_pool = descriptors::create_descriptor_pool(&device, &pool_size);
        let buffer_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(FRAMES_IN_FLIGHT as u32)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::ALL);

        let storage_image_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(300)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL);

        let sample_image_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(300)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL);

        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(5)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .stage_flags(vk::ShaderStageFlags::ALL);

        // setup for bindless descriptors
        let buffer_descriptor_layout =
            descriptors::create_descriptor_layouts(&device, buffer_binding, &[]);
        let storage_descriptor_layout = descriptors::create_descriptor_layouts(
            &device,
            storage_image_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );
        let sample_descriptor_layout = descriptors::create_descriptor_layouts(
            &device,
            sample_image_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );
        let sampler_descriptor_layout = descriptors::create_descriptor_layouts(
            &device,
            sampler_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );

        let buffer_descriptor = descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            buffer_descriptor_layout,
            3,
        );
        let storage_descriptor = descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            storage_descriptor_layout,
            300,
        );
        let sample_descriptor = descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            sample_descriptor_layout,
            300,
        );
        let sampler_descriptor = descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            sampler_descriptor_layout,
            5,
        );

        let descriptor_sets = [
            buffer_descriptor,
            storage_descriptor,
            sample_descriptor,
            sampler_descriptor,
        ];

        let descriptor_layouts = [
            buffer_descriptor_layout,
            storage_descriptor_layout,
            sample_descriptor_layout,
            sampler_descriptor_layout,
        ];

        let push_constant_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(properties.properties.limits.max_push_constants_size)];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(&push_constant_range);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let descriptor =
            descriptors::Descriptor::new(descriptor_pool, descriptor_sets, descriptor_layouts);

        // Prepare frame resources
        let mut frame_resources: [FrameResources; FRAMES_IN_FLIGHT] =
            [FrameResources::default(); FRAMES_IN_FLIGHT];

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_resources[i].command_pool = device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap();
            });
        }

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                let pool = &frame_resources[i].command_pool;

                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                frame_resources[i].command_buffer = device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()[0];
            });
        }

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_resources[i].fence = device.create_fence(&fence_info, None).unwrap();
            });
        }

        let semaphore_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_resources[i].image_acquired_semaphore =
                    device.create_semaphore(&semaphore_info, None).unwrap()
            });
        }

        let render_done_semaphores: Vec<vk::Semaphore> = unsafe {
            (0..swapchain.images.len())
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        let query_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .query_count(MAX_QUERY_COUNT)
            .pipeline_statistics(vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS);

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_resources[i].pipeline_query =
                    device.create_query_pool(&query_info, None).unwrap();
            });
        }

        Self {
            device,
            surface,
            surface_loader,
            physical_device: pdevice,
            instance,
            graphics_queue,
            properties: properties.properties,
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
            frame_resources,
            render_done_semaphores,
            swapchain,
            descriptor,
            pipeline_layout,
            debug_utils_loader,
            debug_callback,
        }
    }

    pub fn update_image_descriptor(
        &self,
        handle: u32,
        view: vk::ImageView,
        tag: RenderResourceTag,
    ) {
        let descriptor = &self.descriptor;

        let info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(view)];

        match tag {
            RenderResourceTag::Storage => {
                let index = get_descriptor_index(RenderResourceTag::Storage);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor.sets[index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(handle)
                    .image_info(&info)
                    .descriptor_count(1);

                let writes = [write];
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }
            }
            RenderResourceTag::Sampled => {
                let index = get_descriptor_index(RenderResourceTag::Sampled);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor.sets[index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_array_element(handle)
                    .image_info(&info)
                    .descriptor_count(1);

                let writes = [write];
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }
            }
            _ => panic!("Incorrect tag"),
        }
    }

    // TODO: register buffer

    pub fn register_samplers(&self, samplers: &[vk::Sampler]) {
        let descriptor = &self.descriptor;

        let infos: Vec<[vk::DescriptorImageInfo; 1]> = samplers
            .iter()
            .map(|sampler| [vk::DescriptorImageInfo::default().sampler(*sampler)])
            .collect();

        let index = get_descriptor_index(RenderResourceTag::Sampler);

        let set = descriptor.sets[index];

        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .dst_array_element(i as _)
                    .image_info(info)
                    .descriptor_count(1)
            })
            .collect();

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn register_image(&self, view: vk::ImageView, tag: RenderResourceTag) -> u32 {
        let descriptor = &self.descriptor;

        let info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(view)];

        match tag {
            RenderResourceTag::Storage => {
                let handle = { descriptor.storage_image_count.lock().unwrap().add() };
                let index = get_descriptor_index(RenderResourceTag::Storage);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor.sets[index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(handle)
                    .image_info(&info)
                    .descriptor_count(1);

                let writes = [write];
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }

                handle
            }
            RenderResourceTag::Sampled => {
                let handle = { descriptor.sample_image_count.lock().unwrap().add() };
                let index = get_descriptor_index(RenderResourceTag::Sampled);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor.sets[index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_array_element(handle)
                    .image_info(&info)
                    .descriptor_count(1);

                let writes = [write];
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }

                handle
            }
            _ => panic!("Incorrect tag"),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let swapchain = &self.swapchain;

            swapchain
                .loader
                .destroy_swapchain(swapchain.swapchain, None);

            self.render_done_semaphores.iter().for_each(|semaphore| {
                self.device.destroy_semaphore(*semaphore, None);
            });

            swapchain.views.iter().for_each(|image_view| {
                self.device.destroy_image_view(*image_view, None);
            });

            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor.destroy(&self.device);

            // dbg!(self.allocator.generate_report());
            ManuallyDrop::drop(&mut self.allocator);

            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// # Safety
pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data;
    unsafe {
        callback_data = *p_callback_data;
    }
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} {message_id_number}] : {message}\n",
    );

    vk::FALSE
}

pub fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
) -> vk::Pipeline {
    let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
        .name(c"main")
        .module(module)
        .stage(vk::ShaderStageFlags::COMPUTE);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .layout(layout)
        .stage(shader_stage_info);

    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .unwrap()
    };

    pipelines[0]
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    shader_stages: &[vk::ShaderStageFlags],
    format: vk::Format,
) -> vk::Pipeline {
    let shader_stage_info: Vec<vk::PipelineShaderStageCreateInfo> = shader_stages
        .iter()
        .map(|stage| match *stage {
            vk::ShaderStageFlags::VERTEX => vk::PipelineShaderStageCreateInfo::default()
                .name(c"vs_main")
                .module(module)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::ShaderStageFlags::FRAGMENT => vk::PipelineShaderStageCreateInfo::default()
                .name(c"ps_main")
                .module(module)
                .stage(vk::ShaderStageFlags::FRAGMENT),
            _ => panic!("Unsupported shader stage"),
        })
        .collect();

    let formats = [format];
    let mut rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .depth_attachment_format(vk::Format::D32_SFLOAT);
    let mut create_flags_info = vk::PipelineCreateFlags2CreateInfoKHR {
        p_next: <*mut _>::cast(&mut rendering_create_info),
        ..Default::default()
    };

    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::COPY)
        .attachments(&attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .min_depth_bounds(1.0)
        .max_depth_bounds(0.0)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
        .depth_test_enable(true)
        .depth_write_enable(true);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .min_sample_shading(1.0)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(vk::CullModeFlags::BACK)
        .line_width(1.0);

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .input_assembly_state(&input_assembly_state)
        .layout(layout)
        .multisample_state(&multisample_state)
        .rasterization_state(&rasterization_state)
        .stages(&shader_stage_info)
        .vertex_input_state(&vertex_input_state)
        .viewport_state(&viewport_state)
        .push_next(&mut create_flags_info);

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .unwrap()
    };

    pipelines[0]
}

#[allow(clippy::manual_div_ceil)]
pub fn get_group_count(size: u32, threads: u32) -> u32 {
    (size + threads - 1) / threads
}

pub fn image_giga_barrier(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image) {
    let image_memory_barrier = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .subresource_range(image_subresource_range(vk::ImageAspectFlags::COLOR))];

    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(&image_memory_barrier);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
}

pub fn giga_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let memory_barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];

    let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barrier);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
}

pub fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    memory_location: gpu_allocator::MemoryLocation,
    flags: vk::BufferUsageFlags,
    size: u64,
) -> Buffer {
    let buffer_info = vk::BufferCreateInfo::default().usage(flags).size(size);
    let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "Buffer allocation",
            requirements,
            location: memory_location,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .unwrap()
    };

    Buffer {
        buffer,
        address: get_buffer_address(device, buffer),
        allocation,
        size,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_buffer_with_data(
    device: &ash::Device,
    queue: Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    allocator: &mut Allocator,
    flags: vk::BufferUsageFlags,
    data: &[u8],
) -> Buffer {
    let size = data.len() as u64;

    let mut staging_buffer = create_buffer(
        device,
        allocator,
        MemoryLocation::CpuToGpu,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        size,
    );
    let ptr = staging_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
    unsafe { ptr.copy_from(data.as_ptr(), data.len()) };

    let buffer = create_buffer(device, allocator, MemoryLocation::GpuOnly, flags, size);

    unsafe {
        device
            .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
            .unwrap()
    }
    let cmd_begin_info = vk::CommandBufferBeginInfo::default();
    unsafe { device.begin_command_buffer(cmd, &cmd_begin_info).unwrap() }
    let copy_region = [vk::BufferCopy2::default().size(size)];

    let copy_buffer_info = &vk::CopyBufferInfo2::default()
        .src_buffer(staging_buffer.buffer)
        .dst_buffer(buffer.buffer)
        .regions(&copy_region);

    unsafe { device.cmd_copy_buffer2(cmd, copy_buffer_info) }

    unsafe { device.end_command_buffer(cmd).unwrap() }
    let command_buffer_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
    let submit_info = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_infos)];

    unsafe {
        device
            .queue_submit2(queue, &submit_info, vk::Fence::null())
            .unwrap()
    }

    unsafe { device.device_wait_idle().unwrap() }

    destroy_buffer(device, allocator, &mut staging_buffer);

    buffer
}

#[allow(clippy::too_many_arguments)]
pub fn create_sampled_image(
    device: &ash::Device,
    queue: Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    allocator: &mut Allocator,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    offsets: &[usize],
    data: &[u8],
    mip: bool,
) -> Image {
    let size = data.len() as u64;

    let mut staging_buffer = create_buffer(
        device,
        allocator,
        MemoryLocation::CpuToGpu,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        size,
    );
    let ptr = staging_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
    unsafe { ptr.copy_from(data.as_ptr(), data.len()) };

    let image = create_image(device, allocator, extent, format, usage, aspect, mip);

    unsafe {
        device
            .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
    }

    let cmd_begin_info = vk::CommandBufferBeginInfo::default();

    unsafe {
        device.begin_command_buffer(cmd, &cmd_begin_info).unwrap();
    }

    image_giga_barrier(device, cmd, image.image);

    let buffer_image_copy: Vec<vk::BufferImageCopy2> = (0..offsets.len())
        .map(|i| {
            let image_subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(i as _);

            vk::BufferImageCopy2::default()
                .image_subresource(image_subresource)
                .image_extent(vk::Extent3D {
                    width: (extent.width >> i).max(1),
                    height: (extent.height >> i).max(1),
                    depth: 1,
                })
                .buffer_offset(offsets[i] as _)
        })
        .collect();

    let copy_buffer_to_image_info = vk::CopyBufferToImageInfo2::default()
        .src_buffer(staging_buffer.buffer)
        .dst_image(image.image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .regions(&buffer_image_copy);

    unsafe {
        device.cmd_copy_buffer_to_image2(cmd, &copy_buffer_to_image_info);
    }

    let command_buffer_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
    let submit_info = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_infos)];

    unsafe {
        device.end_command_buffer(cmd).unwrap();
        device
            .queue_submit2(queue, &submit_info, vk::Fence::null())
            .unwrap()
    }

    unsafe {
        device.device_wait_idle().unwrap();
    }

    destroy_buffer(device, allocator, &mut staging_buffer);

    image
}

pub fn destroy_buffer(device: &ash::Device, allocator: &mut Allocator, buffer: &mut Buffer) {
    unsafe {
        let allocation = std::mem::take(&mut buffer.allocation);
        allocator.free(allocation).unwrap();
        device.destroy_buffer(buffer.buffer, None);
    }
}

pub fn create_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    mip: bool,
) -> Image {
    let mip_levels = if mip {
        extent.width.max(extent.height).ilog2() + 1
    } else {
        1
    };

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .usage(usage)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "Image allocation",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .unwrap()
    };

    let view = create_image_view(
        device,
        image,
        format,
        vk::ImageViewType::TYPE_2D,
        image_subresource_range(aspect),
    );

    Image {
        image,
        view,
        format,
        extent,
        allocation,
    }
}

pub fn destroy_image(device: &ash::Device, allocator: &mut Allocator, image: &mut Image) {
    unsafe {
        let allocation = std::mem::take(&mut image.allocation);
        allocator.free(allocation).unwrap();
        device.destroy_image(image.image, None);
        device.destroy_image_view(image.view, None);
    }
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

pub fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    view_type: vk::ImageViewType,
    subresource_range: vk::ImageSubresourceRange,
) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(subresource_range);

    unsafe { device.create_image_view(&info, None).unwrap() }
}

// Taken from ash
pub fn load_shader(device: &ash::Device, path: &str) -> vk::ShaderModule {
    let mut file = std::fs::File::open(path).expect("Could not read / open file");

    let mut spv: Vec<u8> = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut spv).unwrap();

    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("Failed to read spv file");
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

fn get_buffer_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let buffer_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);

    unsafe { device.get_buffer_device_address(&buffer_address_info) }
}

pub fn get_infinite_reverse_perspective_matrix(fovy: f32, aspect: f32, near: f32) -> glam::Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();

    #[rustfmt::skip]
    let matrix = glam::Mat4::from_cols_array(&[
	    f / aspect, 0.0,  0.0,  0.0,
	           0.0,   f,  0.0,  0.0,
	           0.0, 0.0,  0.0, -1.0,
	           0.0, 0.0, near,  0.0
	]);

    matrix
}

// TODO: bytemuck?
pub fn as_bytes<T>(t: &[T]) -> &[u8] {
    let len = size_of_val(t);
    unsafe { std::slice::from_raw_parts(t.as_ptr() as *const u8, len) }
}

// TODO: bytemuck?
pub fn push_constants_as_bytes<T>(t: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((t as *const T) as *const u8, std::mem::size_of::<T>()) }
}

pub fn create_sampler(
    device: &ash::Device,
    filter: vk::Filter,
    address: vk::SamplerAddressMode,
) -> vk::Sampler {
    let info = vk::SamplerCreateInfo::default()
        .address_mode_u(address)
        .address_mode_v(address)
        .address_mode_w(address)
        .min_filter(filter)
        .mag_filter(filter)
        .max_lod(vk::LOD_CLAMP_NONE);

    unsafe { device.create_sampler(&info, None).unwrap() }
}

pub fn nearest_power_of_two(extent: u32) -> u32 {
    1 << extent.ilog2()
}

pub fn next_power_of_two(extent: u32) -> u32 {
    nearest_power_of_two(extent) << 1
}
