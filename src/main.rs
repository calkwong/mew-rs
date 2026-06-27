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
use std::{borrow::Cow, cell::RefCell, ffi, mem::ManuallyDrop, os::raw::c_char};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const FRAMES_IN_FLIGHT: usize = 1;

#[allow(dead_code)]
struct State {
    window: Window,
    entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    device: Device,
    queue_family_index: u32,
    graphics_queue: Queue,
    allocator: ManuallyDrop<Allocator>,
    image_allocation: Allocation,
    image: vk::Image,
    swapchain_loader: swapchain::Device,
    swapchain: SwapchainKHR,
    swapchain_create_info: vk::SwapchainCreateInfoKHR<'static>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_dirty: RefCell<bool>,
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    debug_utils_loader: debug_utils::Instance,
    debug_callback: DebugUtilsMessengerEXT,
    fences: Vec<Fence>,
    render_done_semaphores: Vec<Semaphore>,
    image_acquired_semaphores: Vec<Semaphore>,
    frame_index: RefCell<usize>,
}

struct App {
    state: Option<State>,
}

impl State {
    fn new(window: Option<Window>) -> Self {
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
        let device_extension_names_raw = [swapchain::NAME.as_ptr()];
        let priorities = [1.0];

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let mut queue_infos: Vec<DeviceQueueCreateInfo> = Vec::new();
        queue_infos.push(queue_create_info);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default();
        let mut vulkan_1_2_features =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let mut vulkan_1_3_features =
            vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);

        // TODO: extension support check

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

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()
        };

        let mut found_format: Option<vk::SurfaceFormatKHR> = None;
        for formats in &surface_formats {
            // if formats.format == vk::Format::B8G8R8A8_UNORM
            if formats.format == vk::Format::B8G8R8A8_SRGB
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

        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let image_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "Example allocation",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false, // TODO: t/f?
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            device
                .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
                .unwrap()
        };

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
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
            (0..swapchain_images.len())
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        let image_acquired_semaphores: Vec<vk::Semaphore> = unsafe {
            (0..FRAMES_IN_FLIGHT)
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        let this = Self {
            window,
            entry,
            instance,
            surface,
            surface_loader,
            device,
            queue_family_index,
            graphics_queue,
            allocator: ManuallyDrop::new(allocator),
            image_allocation,
            image,
            swapchain_loader,
            swapchain,
            swapchain_create_info,
            swapchain_images,
            swapchain_image_views,
            swapchain_extent,
            swapchain_dirty: RefCell::new(false),
            command_pools,
            command_buffers,
            debug_utils_loader,
            debug_callback,
            fences,
            render_done_semaphores,
            image_acquired_semaphores,
            frame_index: RefCell::new(0),
        };

        this
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.swapchain_image_views.iter().for_each(|image_view| {
                self.device.destroy_image_view(*image_view, None);
            });

            self.command_pools.iter().for_each(|pool| {
                self.device.destroy_command_pool(*pool, None);
            });

            self.fences.iter().for_each(|fence| {
                self.device.destroy_fence(*fence, None);
            });

            self.render_done_semaphores.iter().for_each(|semaphore| {
                self.device.destroy_semaphore(*semaphore, None);
            });

            self.image_acquired_semaphores.iter().for_each(|semaphore| {
                self.device.destroy_semaphore(*semaphore, None);
            });

            // clean up allocator + resources
            let allocation = std::mem::take(&mut self.image_allocation);
            self.allocator.free(allocation).unwrap();
            self.device.destroy_image(self.image, None);
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
                    state.swapchain_create_info.image_extent = vk::Extent2D {
                        width: new_size.width,
                        height: new_size.height,
                    };
                    update_swapchain(state);
                    // println!("Resize swapchain driven by WindowEvent::Resized");
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    render_loop(state);

                    if *state.swapchain_dirty.borrow() {
                        let new_size = state.window.inner_size();
                        state.swapchain_create_info.image_extent = vk::Extent2D {
                            width: new_size.width,
                            height: new_size.height,
                        };

                        update_swapchain(state);
                        *state.swapchain_dirty.borrow_mut() = false;
                        // println!("Resize swapchain driven by ERROR_OUT_OF_DATE_KHR");
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
            unsafe { state.device.device_wait_idle().unwrap() };
        }

        self.state = None;
    }
}

fn render_loop(state: &State) {
    let current_index = *state.frame_index.borrow() % FRAMES_IN_FLIGHT;

    let fence = state.fences[current_index];
    unsafe {
        state
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();

        state
            .device
            .reset_command_pool(
                state.command_pools[current_index],
                vk::CommandPoolResetFlags::empty(),
            )
            .unwrap();
    }

    let cmd = state.command_buffers[current_index];
    let cmd_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        state
            .device
            .begin_command_buffer(cmd, &cmd_begin_info)
            .unwrap()
    };

    let acquire_semaphore = state.image_acquired_semaphores[current_index];

    let swapchain_idx: usize;
    unsafe {
        let acquire_result = state.swapchain_loader.acquire_next_image(
            state.swapchain,
            u64::MAX,
            acquire_semaphore,
            vk::Fence::null(),
        );

        match acquire_result {
            Ok((present_idx, _)) => {
                swapchain_idx = present_idx as usize;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let mut swapchain_dirty = state.swapchain_dirty.borrow_mut();
                *swapchain_dirty = true;
                return;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {e:?}");
            }
        }

        state.device.reset_fences(&[fence]).unwrap();
    }

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .image(state.swapchain_images[swapchain_idx])
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
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let image_memory_barrier = [image_memory_barrier];

    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(&image_memory_barrier);

    unsafe { state.device.cmd_pipeline_barrier2(cmd, &dependency_info) };

    unsafe {
        state.device.end_command_buffer(cmd).unwrap();
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
    let swapchain = [state.swapchain];

    let submit_info = vk::SubmitInfo2::default()
        .signal_semaphore_infos(&semaphore_signal_info)
        .wait_semaphore_infos(&semaphore_wait_info)
        .command_buffer_infos(&cmd_submit_info);

    unsafe {
        state
            .device
            .queue_submit2(state.graphics_queue, &[submit_info], fence)
            .unwrap();
    }

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&render_semaphore)
        .image_indices(&swapchain_idx)
        .swapchains(&swapchain);

    unsafe {
        let present_result = state
            .swapchain_loader
            .queue_present(state.graphics_queue, &present_info);

        match present_result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let mut swapchain_dirty = state.swapchain_dirty.borrow_mut();
                *swapchain_dirty = true;
                return;
            }
            Err(e) => {
                panic!("Present error: {e:?}");
            }
        }
    };

    let mut new_frame_index = state.frame_index.borrow_mut();
    *new_frame_index += 1;
}

fn update_swapchain(state: &mut State) {
    unsafe {
        state.device.device_wait_idle().unwrap();
    }

    state
        .swapchain_image_views
        .iter()
        .for_each(|image_view| unsafe {
            state.device.destroy_image_view(*image_view, None);
        });

    state.swapchain_image_views.clear();
    state.swapchain_images.clear();

    let old_swapchain_handle = state.swapchain;
    state.swapchain_create_info.old_swapchain = old_swapchain_handle;

    unsafe {
        state.swapchain = state
            .swapchain_loader
            .create_swapchain(&state.swapchain_create_info, None)
            .unwrap();

        state
            .swapchain_loader
            .destroy_swapchain(old_swapchain_handle, None);
    }

    state.swapchain_images = unsafe {
        state
            .swapchain_loader
            .get_swapchain_images(state.swapchain)
            .unwrap()
    };

    state.swapchain_image_views = state
        .swapchain_images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(state.swapchain_create_info.image_format)
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
                    .device
                    .create_image_view(&image_view_info, None)
                    .unwrap()
            }
        })
        .collect();

    state.swapchain_extent = state.swapchain_create_info.image_extent;
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

/// Untested
#[allow(dead_code)]
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

/// Untested
#[allow(dead_code)]
fn create_descriptor_layouts(
    device: &Device,
    binding: vk::DescriptorSetLayoutBinding,
) -> vk::DescriptorSetLayout {
    let binding = [binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

/// Untested
#[allow(dead_code)]
fn create_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> vk::DescriptorSet {
    let layout = [layout];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layout);

    unsafe {
        device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap()[0]
    }
}

/// Untested
#[allow(dead_code)]
fn create_compute_pipeline(device: Device, layout: vk::PipelineLayout, module: vk::ShaderModule) -> vk::Pipeline {
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

/// Untested
#[allow(dead_code)]
fn write_descriptor_sets() {}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { state: None }).unwrap();
}
