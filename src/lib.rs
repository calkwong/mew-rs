#![allow(dead_code)]

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{self, DebugUtilsMessengerEXT, Queue, SwapchainKHR},
};
use ash_window;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::*;
use std::{borrow::Cow, ffi, mem::ManuallyDrop, os::raw::c_char};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

pub mod descriptors;

pub struct Swapchain {
    pub loader: swapchain::Device,
    pub swapchain: SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub dirty: bool,
}

pub struct Engine {
    pub entry: Entry,
    pub instance: Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue_family_index: u32,
    pub graphics_queue: Queue,
    pub allocator: ManuallyDrop<Allocator>,
    pub swapchain: Swapchain,
    pub debug_utils_loader: debug_utils::Instance,
    pub debug_callback: DebugUtilsMessengerEXT,
}

impl Engine {
    pub fn new(window: Option<&Window>) -> Self {
        let entry = Entry::linked();

        let window = window.unwrap();

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

        let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
        queue_infos.push(queue_create_info);

        // check extension support
        let device_extension_names_raw = [swapchain::NAME.as_ptr()];

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

        assert!(enabled_vulkan_1_2_features.buffer_device_address > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_partially_bound > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_variable_descriptor_count > 0);
        assert!(enabled_vulkan_1_2_features.runtime_descriptor_array > 0);
        assert!(enabled_vulkan_1_3_features.synchronization2 > 0);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default();
        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true);
        let mut vulkan_1_3_features =
            vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&vulkan_1_0_features)
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

        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let swapchain_format = get_swapchain_format(&surface_loader, pdevice, surface);

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap()
        };

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }

        let window_size = window.inner_size();

        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let present_mode = get_present_mode(&surface_loader, pdevice, surface);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_format(swapchain_format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|image| {
                let image_view_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&image_view_info, None).unwrap() }
            })
            .collect();

        Self {
            entry,
            instance,
            surface,
            surface_loader,
            physical_device: pdevice,
            device,
            queue_family_index,
            graphics_queue,
            allocator: ManuallyDrop::new(allocator),
            swapchain: Swapchain {
                loader: swapchain_loader,
                swapchain: swapchain,
                format: swapchain_format,
                extent: surface_resolution,
                images: swapchain_images,
                views: swapchain_image_views,
                dirty: false,
            },
            debug_utils_loader,
            debug_callback,
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            let swapchain = &self.swapchain;

            swapchain
                .loader
                .destroy_swapchain(swapchain.swapchain, None);

            swapchain.views.iter().for_each(|image_view| {
                self.device.destroy_image_view(*image_view, None);
            });

            dbg!(self.allocator.generate_report());
            ManuallyDrop::drop(&mut self.allocator);

            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

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
    device: &Device,
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

pub fn push_constants<T>(
    device: &Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    data: &T,
) {
    let size = std::mem::size_of::<T>();
    let len = size / std::mem::size_of::<u8>();

    let as_u8_slice = unsafe { std::slice::from_raw_parts((data as *const T) as *const u8, len) };

    let mut constants = [0u8; 256];
    constants[..size].copy_from_slice(as_u8_slice);

    unsafe {
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::ALL, 0, &constants);
    }
}

pub fn get_group_count(size: u32, threads: u32) -> u32 {
    (size + threads - 1) / threads
}

pub fn giga_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let memory_barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];

    let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barrier);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
}

// TODO: refactor
pub fn create_image(
    device: &Device,
    allocator: &mut Allocator,
    extent: vk::Extent2D,
) -> (vk::Image, Allocation) {
    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .samples(vk::SampleCountFlags::TYPE_1);

    let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "Example allocation",
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

    (image, allocation)
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect_mask,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

pub fn create_image_view(
    device: &Device,
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

pub fn load_shader(device: &Device, path: &str) -> vk::ShaderModule {
    let mut file = std::fs::File::open(path).expect("Could not read / open file");

    let mut spv: Vec<u8> = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut spv).unwrap();

    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("Failed to read spv file");
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

pub fn get_swapchain_format(
    surface_loader: &surface::Instance,
    pdevice: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::Format {
    let surface_formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap()
    };

    for formats in &surface_formats {
        if formats.format == vk::Format::B8G8R8A8_UNORM
        // if formats.format == vk::Format::B8G8R8A8_SRGB // this does not support STORAGE usage
        {
            return vk::Format::B8G8R8A8_UNORM;
        }
    }

    surface_formats[0].format
}

pub fn get_present_mode(
    surface_loader: &surface::Instance,
    pdevice: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::PresentModeKHR {
    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(pdevice, surface)
            .unwrap()
    };

    present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}
