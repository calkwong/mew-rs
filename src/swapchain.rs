use ash::{
    Instance,
    khr::{surface, swapchain},
    vk::{self, SwapchainKHR},
};

use crate as mew;
use mew::Device;

use winit::window::Window;

pub struct Swapchain {
    pub loader: swapchain::Device,
    pub swapchain: SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub dirty: bool,
}

pub fn create_swapchain(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    device: &ash::Device,
    window: &Window,
    old_handle: vk::SwapchainKHR,
) -> Swapchain {
    let swapchain_loader = swapchain::Device::new(instance, device);

    let swapchain_format = get_swapchain_format(surface_loader, pdevice, surface);

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

    let present_mode = get_present_mode(surface_loader, pdevice, surface);

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_format(swapchain_format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    if old_handle != vk::SwapchainKHR::null() {
        swapchain_create_info = swapchain_create_info.old_swapchain(old_handle);
    }

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

    Swapchain {
        loader: swapchain_loader,
        swapchain,
        format: swapchain_format,
        extent: surface_resolution,
        images: swapchain_images,
        views: swapchain_image_views,
        dirty: false,
    }
}

pub fn recreate_swapchain(device: &mut Device, window: &Window) {
    unsafe {
        device.device.device_wait_idle().unwrap();
    }

    unsafe {
        for view in &device.swapchain.views {
            device.device.destroy_image_view(*view, None);
        }
    }

    let old_handle = device.swapchain.swapchain;

    device.swapchain = create_swapchain(
        &device.instance,
        &device.surface_loader,
        device.surface,
        device.physical_device,
        &device.device,
        window,
        old_handle,
    );

    unsafe {
        device.swapchain.loader.destroy_swapchain(old_handle, None);
    }
}

fn get_swapchain_format(
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

fn get_present_mode(
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
    // .unwrap_or(vk::PresentModeKHR::IMMEDIATE)
}
