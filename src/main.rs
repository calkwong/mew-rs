use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{self, DeviceQueueCreateInfo},
};
use ash_window;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

use std::{borrow::Cow, ffi, os::raw::c_char};

struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_width: u32 = 1700;
        let window_height: u32 = 900;

        if let None = self.window {
            self.window = Some(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("window: mew-rust")
                            .with_inner_size(winit::dpi::LogicalSize::new(
                                f64::from(1700),
                                f64::from(900),
                            )),
                    )
                    .unwrap(),
            );

            // TODO: we don't need this entire block in unsafe
            // TODO: place instance, device, surface etc. in a new container
            unsafe {
                let entry = Entry::linked();

                let window_handle = self
                    .window
                    .as_ref()
                    .unwrap()
                    .window_handle()
                    .expect("Window handle error")
                    .as_raw();

                let app_name = c"mew-rust";

                let layer_names = [c"VK_LAYER_KHRONOS_validation"];
                let layers_names_raw: Vec<*const c_char> = layer_names
                    .iter()
                    .map(|raw_name| raw_name.as_ptr())
                    .collect();

                let display_handle = event_loop
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

                let instance: Instance = entry
                    .create_instance(&instance_create_info, None)
                    .expect("Instance creation failed");

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
                let debug_call_back = debug_utils_loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap();

                let surface = ash_window::create_surface(
                    &entry,
                    &instance,
                    display_handle,
                    window_handle,
                    None,
                )
                .unwrap();
                let surface_loader = surface::Instance::new(&entry, &instance);
                let pdevices = instance
                    .enumerate_physical_devices()
                    .expect("Physical device error");

                let (pdevice, queue_family_index) = pdevices
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
                                if supports_graphic_and_surface {
                                    Some((*pdevice, index))
                                } else {
                                    None
                                }
                            })
                    })
                    .expect("Couldn't find suitable device");
                let queue_family_index = queue_family_index as u32;
                let device_extension_names_raw = [swapchain::NAME.as_ptr()];
                let features = vk::PhysicalDeviceFeatures::default();
                let priorities = [1.0];

                let queue_create_info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&priorities); // TODO

                let mut queue_infos: Vec<DeviceQueueCreateInfo> = Vec::new();
                queue_infos.push(queue_create_info);

                // TODO: features and extensions
                let device_create_info = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&device_extension_names_raw)
                    .enabled_features(&features);

                let device = instance
                    .create_device(pdevice, &device_create_info, None)
                    .unwrap();

                let _present_queue = device.get_device_queue(queue_family_index, 0);

                // TODO: do we want to be selecting a surface format here?
                let surface_format = surface_loader
                    .get_physical_device_surface_formats(pdevice, surface)
                    .unwrap()[0];

                let surface_capabilities = surface_loader
                    .get_physical_device_surface_capabilities(pdevice, surface)
                    .unwrap();

                // TODO: cross reference mew-engine
                let mut desired_image_count = surface_capabilities.min_image_count + 1;
                if surface_capabilities.max_image_count > 0
                    && desired_image_count > surface_capabilities.max_image_count
                {
                    desired_image_count = surface_capabilities.max_image_count;
                }
                let surface_resolution = match surface_capabilities.current_extent.width {
                    u32::MAX => vk::Extent2D {
                        width: window_width,
                        height: window_height,
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
                let present_modes = surface_loader
                    .get_physical_device_surface_present_modes(pdevice, surface)
                    .unwrap();
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
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE) //
                    .pre_transform(pre_transform) //
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) //
                    .present_mode(present_mode)
                    .clipped(true) //
                    .image_array_layers(1); //

                let swapchain = swapchain_loader
                    .create_swapchain(&swapchain_create_info, None)
                    .unwrap();

                let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
                let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
                    .iter()
                    .map(|&image| {
                        let image_view_info = vk::ImageViewCreateInfo::default()
                            .image(image)
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
                                layer_count: 1
                            });
                        device.create_image_view(&image_view_info, None).unwrap()
                    })
                    .collect();

                // TODO: move to exiting() and impl Drop
                swapchain_loader.destroy_swapchain(swapchain, None);
                swapchain_image_views.iter().for_each(|image_view| {
                    device.destroy_image_view(*image_view, None);
                });
                surface_loader.destroy_surface(surface, None);
                device.destroy_device(None);
                debug_utils_loader.destroy_debug_utils_messenger(debug_call_back, None);
                instance.destroy_instance(None);

                println!("Does this execute once only?");
            }
        }
    }

    // TODO: refactor when we have actual window, currently force exit required
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
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
            _ => {
                println!("Looping forever");
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.run_app(&mut App { window: None }).unwrap();

    println!("Exiting app");
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
