use ash::vk::{self, Fence, Semaphore};
use gpu_allocator::vulkan::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const FRAMES_IN_FLIGHT: usize = 1;

struct State {
    window: Window,
    engine: mew::Engine,
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

impl State {
    fn new(window: Option<Window>) -> Self {
        // TODO: i don't like this mut for the sake of image creation
        let mut engine = mew::Engine::new(window.as_ref());

        let device = &engine.device;

        let (draw_image, image_allocation) =
            mew::create_image(&device, &mut engine.allocator, engine.swapchain_extent);

        let draw_image_view = mew::create_image_view(
            device,
            draw_image,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageViewType::TYPE_2D,
            mew::image_subresource_range(vk::ImageAspectFlags::COLOR),
        );

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
        let descriptor_pool = mew::descriptors::create_descriptor_pool(&device, pool_size);
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(10)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL);

        // this is part of bindless descriptors
        let descriptor_set_layout = mew::descriptors::create_descriptor_layouts(&device, binding);
        let storage_image_descriptor_counts = engine.swapchain_images.len() as u32 + 1;
        let descriptor_set = mew::descriptors::create_descriptor_sets(
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
        let color_shader_module = mew::load_shader(device, "shaders/compiled/color.spv");
        let copy_shader_module = mew::load_shader(device, "shaders/compiled/copy_swapchain.spv");

        let color_pipeline =
            mew::create_compute_pipeline(&device, pipeline_layout, color_shader_module);
        let copy_pipeline =
            mew::create_compute_pipeline(&device, pipeline_layout, copy_shader_module);

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

struct App {
    state: Option<State>,
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
                    (state.draw_image, state.image_allocation) = mew::create_image(
                        &device,
                        &mut state.engine.allocator,
                        state.engine.swapchain_extent,
                    );

                    state.draw_image_view = mew::create_image_view(
                        device,
                        state.draw_image,
                        vk::Format::R16G16B16A16_SFLOAT,
                        vk::ImageViewType::TYPE_2D,
                        mew::image_subresource_range(vk::ImageAspectFlags::COLOR),
                    );

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
        mew::push_constants(&device, cmd, state.pipeline_layout, &pc);
        let group_count_x = mew::get_group_count(state.engine.swapchain_extent.width, 8);
        let group_count_y = mew::get_group_count(state.engine.swapchain_extent.height, 8);
        device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    };

    mew::giga_barrier(&device, cmd);

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
        mew::push_constants(&device, cmd, state.pipeline_layout, &pc);
        let group_count_x = mew::get_group_count(state.engine.swapchain_extent.width, 8);
        let group_count_y = mew::get_group_count(state.engine.swapchain_extent.height, 8);
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
            mew::create_image_view(
                &state.engine.device,
                image,
                state.engine.swapchain_create_info.image_format,
                vk::ImageViewType::TYPE_2D,
                mew::image_subresource_range(vk::ImageAspectFlags::COLOR),
            )
        })
        .collect();

    state.engine.swapchain_extent = state.engine.swapchain_create_info.image_extent;
}

fn main() {
    /* unsafe {
        std::env::remove_var("WAYLAND_DISPLAY");
    } */

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { state: None }).unwrap();
}
