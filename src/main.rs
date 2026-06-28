use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{
        self, DebugUtilsMessengerEXT, DeviceQueueCreateInfo, Fence, Queue, Semaphore, SwapchainKHR,
    },
};
use ash_window;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::*;
use std::{borrow::Cow, ffi, io::Cursor, mem::ManuallyDrop, os::raw::c_char};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const FRAMES_IN_FLIGHT: usize = 1;

#[allow(dead_code)]
struct Engine {
    entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    device: Device,
    queue_family_index: u32,
    graphics_queue: Queue,
    allocator: ManuallyDrop<Allocator>,
    swapchain_loader: swapchain::Device,
    swapchain: SwapchainKHR,
    swapchain_create_info: vk::SwapchainCreateInfoKHR<'static>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_dirty: bool,
    debug_utils_loader: debug_utils::Instance,
    debug_callback: DebugUtilsMessengerEXT,
}

impl Engine {
    fn new(window: Option<&Window>) -> Self {
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

        let mut queue_infos: Vec<DeviceQueueCreateInfo> = Vec::new();
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

        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()
        };

        let mut found_format: Option<vk::SurfaceFormatKHR> = None;
        for formats in &surface_formats {
            if formats.format == vk::Format::B8G8R8A8_UNORM
            // if formats.format == vk::Format::B8G8R8A8_SRGB // this does not support STORAGE usage
                && formats.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                found_format = Some(*formats);
                break;
            }
        }

        let surface_format = if let Some(format) = found_format {
            format
        } else {
            surface_formats[0]
        };

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
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap()
        };
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
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
                    .format(surface_format.format)
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

        let swapchain_extent = surface_resolution;

        Self {
            entry,
            instance,
            surface,
            surface_loader,
            device,
            queue_family_index,
            graphics_queue,
            allocator: ManuallyDrop::new(allocator),
            swapchain_loader,
            swapchain,
            swapchain_create_info,
            swapchain_images,
            swapchain_image_views,
            swapchain_extent,
            swapchain_dirty: false,
            debug_utils_loader,
            debug_callback,
        }
    }
}

#[allow(dead_code)]
struct State {
    window: Window,
    engine: Engine,
    image_allocation: Allocation,
    draw_image: vk::Image,
    draw_image_view: vk::ImageView,
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    fences: Vec<Fence>,
    render_done_semaphores: Vec<Semaphore>,
    image_acquired_semaphores: Vec<Semaphore>,
    frame_index: usize,
    color_pipeline: vk::Pipeline,
    copy_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

struct App {
    state: Option<State>,
}

impl State {
    fn new(window: Option<Window>) -> Self {
        // i don't like this mut for the sake of image creation
        let mut engine = Engine::new(window.as_ref());

        let device = &engine.device;

        let (draw_image, image_allocation) =
            create_image(&device, &mut engine.allocator, engine.swapchain_extent);

        // TODO: create helper fn, and also create a Image container that holds allocation, vk::Image etc.
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(draw_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
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
        let draw_image_view = unsafe { device.create_image_view(&image_view_info, None).unwrap() };

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(engine.queue_family_index);
        let command_pools: Vec<vk::CommandPool> = unsafe {
            (0..FRAMES_IN_FLIGHT)
                .map(|_| {
                    device
                        .create_command_pool(&command_pool_info, None)
                        .unwrap()
                })
                .collect()
        };

        let command_buffers: Vec<vk::CommandBuffer> = command_pools
            .iter()
            .map(|pool| unsafe {
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                // note: a single command buffer per command pool
                device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()[0]
            })
            .collect();

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let fences = unsafe {
            (0..FRAMES_IN_FLIGHT)
                .map(|_| device.create_fence(&fence_info, None).unwrap())
                .collect()
        };

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let render_done_semaphores: Vec<vk::Semaphore> = unsafe {
            (0..engine.swapchain_images.len())
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        let image_acquired_semaphores: Vec<vk::Semaphore> = unsafe {
            (0..FRAMES_IN_FLIGHT)
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        // TODO: descriptors
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 100,
        };
        let descriptor_pool = create_descriptor_pool(&device, pool_size);
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(10)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL);

        // this is part of bindless descriptors
        let descriptor_set_layout = create_descriptor_layouts(&device, binding);
        let storage_image_descriptor_counts = engine.swapchain_images.len() as u32 + 1;
        let descriptor_set = create_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            storage_image_descriptor_counts,
        );

        // TODO: create fn for write desc set
        let mut image_infos: Vec<vk::DescriptorImageInfo> =
            Vec::from([vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(draw_image_view)]);
        engine.swapchain_image_views.iter().for_each(|image_view| {
            image_infos.push(
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(*image_view),
            );
        });
        let storage_descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos)
            .descriptor_count(storage_image_descriptor_counts);
        unsafe {
            device.update_descriptor_sets(&[storage_descriptor_write], &[]);
        }

        // TODO: create fn for this
        let descriptor_set_layouts = [descriptor_set_layout];
        let push_constant_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            // .size(properties.properties.limits.max_push_constants_size)];
            .size(256)]; // TODO: hardcoded
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_range);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        // taken from ash
        let mut spv_file = Cursor::new(&include_bytes!("../shaders/compiled/color.spv")[..]);
        let code = ash::util::read_spv(&mut spv_file).expect("Failed to read spv file");
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let color_shader_module =
            unsafe { device.create_shader_module(&create_info, None).unwrap() };

        let mut spv_file =
            Cursor::new(&include_bytes!("../shaders/compiled/copy_swapchain.spv")[..]);
        let code = ash::util::read_spv(&mut spv_file).expect("Failed to read spv file");
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let copy_shader_module =
            unsafe { device.create_shader_module(&create_info, None).unwrap() };

        let color_pipeline = create_compute_pipeline(&device, pipeline_layout, color_shader_module);
        let copy_pipeline = create_compute_pipeline(&device, pipeline_layout, copy_shader_module);

        unsafe {
            device.destroy_shader_module(color_shader_module, None);
            device.destroy_shader_module(copy_shader_module, None);
        }

        let this = Self {
            window: window.unwrap(),
            engine,
            image_allocation,
            draw_image,
            draw_image_view,
            command_pools,
            command_buffers,
            fences,
            render_done_semaphores,
            image_acquired_semaphores,
            frame_index: 0,
            color_pipeline,
            copy_pipeline,
            pipeline_layout,
            descriptor_set,
            descriptor_pool,
            descriptor_set_layout,
        };

        this
    }
}

impl Drop for State {
    fn drop(&mut self) {
        println!("dropping state");
        let device = &self.engine.device;

        unsafe {
            self.command_pools.iter().for_each(|pool| {
                device.destroy_command_pool(*pool, None);
            });

            self.fences.iter().for_each(|fence| {
                device.destroy_fence(*fence, None);
            });

            self.render_done_semaphores.iter().for_each(|semaphore| {
                device.destroy_semaphore(*semaphore, None);
            });

            self.image_acquired_semaphores.iter().for_each(|semaphore| {
                device.destroy_semaphore(*semaphore, None);
            });

            // clean up allocator + resources
            let allocation = std::mem::take(&mut self.image_allocation);
            self.engine.allocator.free(allocation).unwrap();
            device.destroy_image(self.draw_image, None);
            device.destroy_image_view(self.draw_image_view, None);

            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.color_pipeline, None);
            device.destroy_pipeline(self.copy_pipeline, None);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        println!("dropping engine");
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.swapchain_image_views.iter().for_each(|image_view| {
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(_) = self.state.as_ref() {
            return;
        }

        let window_width: u32 = 1700;
        let window_height: u32 = 900;

        let window = Some(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("window: mew-rust")
                        .with_resizable(true)
                        .with_inner_size(winit::dpi::LogicalSize::new(
                            f64::from(window_width),
                            f64::from(window_height),
                        )),
                )
                .unwrap(),
        );

        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                if let Some(state) = self.state.as_mut() {
                    state.engine.swapchain_create_info.image_extent = vk::Extent2D {
                        width: new_size.width,
                        height: new_size.height,
                    };

                    update_swapchain(state);

                    let device = &state.engine.device;

                    // destroy outdated resources
                    unsafe {
                        let allocation = std::mem::take(&mut state.image_allocation);
                        state.engine.allocator.free(allocation).unwrap();
                        device.destroy_image(state.draw_image, None);
                        device.destroy_image_view(state.draw_image_view, None);
                    }

                    // update resources
                    (state.draw_image, state.image_allocation) = create_image(
                        &device,
                        &mut state.engine.allocator,
                        state.engine.swapchain_extent,
                    );

                    let image_view_info = vk::ImageViewCreateInfo::default()
                        .image(state.draw_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
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
                    state.draw_image_view =
                        unsafe { device.create_image_view(&image_view_info, None).unwrap() };

                    let mut image_infos: Vec<vk::DescriptorImageInfo> =
                        Vec::from([vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(state.draw_image_view)]);
                    state
                        .engine
                        .swapchain_image_views
                        .iter()
                        .for_each(|image_view| {
                            image_infos.push(
                                vk::DescriptorImageInfo::default()
                                    .image_layout(vk::ImageLayout::GENERAL)
                                    .image_view(*image_view),
                            );
                        });
                    let storage_image_descriptor_counts =
                        state.engine.swapchain_images.len() as u32 + 1;
                    let storage_descriptor_write = vk::WriteDescriptorSet::default()
                        .dst_set(state.descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_infos)
                        .descriptor_count(storage_image_descriptor_counts);
                    unsafe {
                        device.update_descriptor_sets(&[storage_descriptor_write], &[]);
                    }

                    println!("Resize swapchain driven by WindowEvent::Resized");
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    render_loop(state);

                    if state.engine.swapchain_dirty {
                        let new_size = state.window.inner_size();
                        if new_size.width == state.engine.swapchain_extent.width
                            && new_size.height == state.engine.swapchain_extent.height
                        {
                            state.engine.swapchain_dirty = false;
                            return;
                        }

                        // this definitely panics on x11
                        todo!("Handle legitimate OUT_OF_DATE_KHR or SUBOPTIMAL_KHR");
                    }
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            unsafe { state.engine.device.device_wait_idle().unwrap() };
        }

        self.state = None;
    }
}

fn render_loop(state: &mut State) {
    let device = &state.engine.device;
    let current_index = state.frame_index % FRAMES_IN_FLIGHT;

    let fence = state.fences[current_index];
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();

        device
            .reset_command_pool(
                state.command_pools[current_index],
                vk::CommandPoolResetFlags::empty(),
            )
            .unwrap();
    }

    let cmd = state.command_buffers[current_index];
    let cmd_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(cmd, &cmd_begin_info).unwrap() };

    unsafe {
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            state.pipeline_layout,
            0,
            &[state.descriptor_set],
            &[],
        );
    }

    let acquire_semaphore = state.image_acquired_semaphores[current_index];

    let swapchain_idx: usize;
    unsafe {
        let acquire_result = state.engine.swapchain_loader.acquire_next_image(
            state.engine.swapchain,
            u64::MAX,
            acquire_semaphore,
            vk::Fence::null(),
        );

        match acquire_result {
            Ok((present_idx, _)) => {
                swapchain_idx = present_idx as usize;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                state.engine.swapchain_dirty = true;
                return;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {e:?}");
            }
        }

        device.reset_fences(&[fence]).unwrap();
    }

    let mut image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .image(state.engine.swapchain_images[swapchain_idx])
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL);

    let mut image_memory_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
    image_memory_barriers.push(image_memory_barrier);
    // draw image
    image_memory_barrier.image = state.draw_image;
    image_memory_barrier.new_layout = vk::ImageLayout::GENERAL;
    image_memory_barriers.push(image_memory_barrier);

    let mut dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.color_pipeline);

        #[allow(dead_code)]
        struct PushConstants {
            draw_id: u32,
        }
        let pc = PushConstants { draw_id: 0 };
        push_constants(&device, cmd, state.pipeline_layout, &pc);
        let group_count_x = get_group_count(state.engine.swapchain_extent.width, 8);
        let group_count_y = get_group_count(state.engine.swapchain_extent.height, 8);
        device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    };

    giga_barrier(&device, cmd);

    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.copy_pipeline);
        #[allow(dead_code)]
        struct PushConstants {
            src_id: u32,
            dst_id: u32,
        }
        let pc = PushConstants {
            src_id: 0,
            dst_id: swapchain_idx as u32 + 1,
        };
        push_constants(&device, cmd, state.pipeline_layout, &pc);
        let group_count_x = get_group_count(state.engine.swapchain_extent.width, 8);
        let group_count_y = get_group_count(state.engine.swapchain_extent.height, 8);
        device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    }

    // transition to present
    image_memory_barrier.image = state.engine.swapchain_images[swapchain_idx];
    image_memory_barrier.old_layout = vk::ImageLayout::GENERAL;
    image_memory_barrier.new_layout = vk::ImageLayout::PRESENT_SRC_KHR;
    image_memory_barriers.clear();
    image_memory_barriers.push(image_memory_barrier);

    dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

    unsafe {
        device.end_command_buffer(cmd).unwrap();
    }

    let cmd_submit_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);

    let semaphore_wait_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(acquire_semaphore)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);

    let render_semaphore = state.render_done_semaphores[swapchain_idx];
    let semaphore_signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(render_semaphore)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);

    let render_semaphore = [render_semaphore];
    let semaphore_signal_info = [semaphore_signal_info];
    let semaphore_wait_info = [semaphore_wait_info];
    let cmd_submit_info = [cmd_submit_info];
    let swapchain_idx = [swapchain_idx as u32];
    let swapchain = [state.engine.swapchain];

    let submit_info = vk::SubmitInfo2::default()
        .signal_semaphore_infos(&semaphore_signal_info)
        .wait_semaphore_infos(&semaphore_wait_info)
        .command_buffer_infos(&cmd_submit_info);

    unsafe {
        device
            .queue_submit2(state.engine.graphics_queue, &[submit_info], fence)
            .unwrap();
    }

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&render_semaphore)
        .image_indices(&swapchain_idx)
        .swapchains(&swapchain);

    unsafe {
        let present_result = state
            .engine
            .swapchain_loader
            .queue_present(state.engine.graphics_queue, &present_info);

        match present_result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                state.engine.swapchain_dirty = true;
                return;
            }
            Err(e) => {
                panic!("Present error: {e:?}");
            }
        }
    };

    state.frame_index += 1;
}

fn update_swapchain(state: &mut State) {
    let device = &state.engine.device;

    unsafe {
        device.device_wait_idle().unwrap();
    }

    state
        .engine
        .swapchain_image_views
        .iter()
        .for_each(|image_view| unsafe {
            device.destroy_image_view(*image_view, None);
        });

    state.engine.swapchain_image_views.clear();
    state.engine.swapchain_images.clear();

    let old_swapchain_handle = state.engine.swapchain;
    state.engine.swapchain_create_info.old_swapchain = old_swapchain_handle;
    unsafe {
        state.engine.swapchain = state
            .engine
            .swapchain_loader
            .create_swapchain(&state.engine.swapchain_create_info, None)
            .unwrap();

        state
            .engine
            .swapchain_loader
            .destroy_swapchain(old_swapchain_handle, None);
    }

    state.engine.swapchain_images = unsafe {
        state
            .engine
            .swapchain_loader
            .get_swapchain_images(state.engine.swapchain)
            .unwrap()
    };

    state.engine.swapchain_image_views = state
        .engine
        .swapchain_images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(state.engine.swapchain_create_info.image_format)
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
            unsafe {
                state
                    .engine
                    .device
                    .create_image_view(&image_view_info, None)
                    .unwrap()
            }
        })
        .collect();

    state.engine.swapchain_extent = state.engine.swapchain_create_info.image_extent;
}

unsafe extern "system" fn vulkan_debug_callback(
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

fn create_descriptor_pool(
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

fn create_descriptor_layouts(
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

fn create_descriptor_sets(
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

fn create_compute_pipeline(
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

fn push_constants<T>(
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

fn get_group_count(size: u32, threads: u32) -> u32 {
    (size + threads - 1) / threads
}

fn giga_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let memory_barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];

    let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barrier);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
}

// TODO: refactor
fn create_image(
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

fn main() {
    /* unsafe {
        std::env::remove_var("WAYLAND_DISPLAY");
    } */

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { state: None }).unwrap();
}
