use ash::vk::{self, Fence};
use glam::Vec4Swizzles;
use mew::basic_renderer;
use mew::descriptors::RenderResourceTag;
use mew::rendergraph::{Rendergraph, Pass};
use mew::swapchain::recreate_swapchain;
use mew::{
    camera::{Camera, Key, KeyState},
    giga_barrier,
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey::Code},
    window::{Window, WindowId},
};

const CURRENT_QUERIES: u32 = 1;

struct Renderer {
    window: Window,
    backend: mew::Device,

    camera: Camera,
    last_frame_time: Option<Instant>,
    frame_index: usize,

    copy_pipeline: vk::Pipeline,
    draw_pipeline: vk::Pipeline,
    cull_pipeline: vk::Pipeline,

    framebuffer: Framebuffer,

    vertex_buffer: mew::Buffer,
    index_buffer: mew::Buffer,
    mesh_buffer: mew::Buffer,
    object_buffer: mew::Buffer,
    draw_indirect_buffer: mew::Buffer,
    dispatch_buffer: mew::Buffer,
}

struct Framebuffer {
    draw_image: mew::Image,
    depth_image: mew::Image,
    draw_index: u32,
    swapchain_indices: Vec<u32>,
}

impl Renderer {
    fn new(window: Window) -> Self {
        let backend = mew::Device::new(&window);

        let camera = Camera::default().position(glam::Vec3::new(0.0, 0.0, 5.0));

        let device = &backend.device;

        let allocator = (*backend.allocator).clone();
        let mut allocator = &mut allocator.lock().unwrap();

        let draw_image = mew::create_image(
            &device,
            &mut allocator,
            backend.swapchain.extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let depth_image = mew::create_image(
            &device,
            &mut allocator,
            backend.swapchain.extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            false,
        );

        let draw_index = backend.register_image(draw_image.view, RenderResourceTag::Storage);

        let swapchain_indices: Vec<u32> = backend
            .swapchain
            .views
            .iter()
            .map(|view| backend.register_image(*view, RenderResourceTag::Storage))
            .collect();

        let copy_shader = mew::load_shader(device, "shaders/compiled/copy_swapchain.spv");
        let draw_shader = mew::load_shader(device, "shaders/compiled/mesh.spv");
        let cull_shader = mew::load_shader(device, "shaders/compiled/culling.spv");

        let cull_pipeline =
            mew::create_compute_pipeline(&device, backend.pipeline_layout, cull_shader);
        let copy_pipeline =
            mew::create_compute_pipeline(&device, backend.pipeline_layout, copy_shader);
        let draw_pipeline = mew::create_graphics_pipeline(
            &device,
            backend.pipeline_layout,
            draw_shader,
            &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
            draw_image.format,
        );

        unsafe {
            device.destroy_shader_module(copy_shader, None);
            device.destroy_shader_module(draw_shader, None);
            device.destroy_shader_module(cull_shader, None);
        }

        let gltf_path = std::env::args().nth(1).unwrap();
        let scene = mew::loader::load_gltf(&gltf_path);

        let vertices = mew::as_bytes(&scene.vertices);
        let vertex_buffer = mew::create_buffer_with_data(
            device,
            backend.graphics_queue,
            backend.frame_resources[0].command_pool,
            backend.frame_resources[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vertices.len() as u64,
            vertices,
        );

        let indices = mew::as_bytes(&scene.indices);
        let index_buffer = mew::create_buffer_with_data(
            device,
            backend.graphics_queue,
            backend.frame_resources[0].command_pool,
            backend.frame_resources[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDEX_BUFFER,
            indices.len() as u64,
            indices,
        );

        let meshes = mew::as_bytes(&scene.meshes);
        let mesh_buffer = mew::create_buffer_with_data(
            device,
            backend.graphics_queue,
            backend.frame_resources[0].command_pool,
            backend.frame_resources[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            meshes.len() as u64,
            meshes,
        );

        let objects = mew::as_bytes(&scene.renderables);
        let object_buffer = mew::create_buffer_with_data(
            device,
            backend.graphics_queue,
            backend.frame_resources[0].command_pool,
            backend.frame_resources[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            objects.len() as u64,
            objects,
        );

        let draw_indirect_buffer = mew::create_buffer(
            device,
            &mut allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            (scene.renderables.len() * std::mem::size_of::<vk::DrawIndexedIndirectCommand>())
                as u64,
        );

        let dispatch_buffer = mew::create_buffer(
            device,
            &mut allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            (3 * std::mem::size_of::<u32>()) as u64,
        );

        let this = Self {
            window,
            backend,
            camera,
            last_frame_time: None,
            frame_index: 0,
            copy_pipeline,
            draw_pipeline,
            cull_pipeline,
            framebuffer: Framebuffer {
                draw_image,
                depth_image,
                draw_index,
                swapchain_indices,
            },
            vertex_buffer,
            index_buffer,
            mesh_buffer,
            object_buffer,
            draw_indirect_buffer,
            dispatch_buffer,
        };

        this
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.backend.device;

        unsafe {
            self.backend.frame_resources.iter().for_each(|data| {
                device.destroy_command_pool(data.command_pool, None);
                device.destroy_fence(data.fence, None);
                device.destroy_semaphore(data.image_acquired_semaphore, None);
                device.destroy_query_pool(data.pipeline_query, None);
            });

            let mut allocator = self.backend.allocator.lock().unwrap();

            // Destroy resources
            mew::destroy_image(device, &mut allocator, &mut self.framebuffer.draw_image);
            mew::destroy_image(device, &mut allocator, &mut self.framebuffer.depth_image);
            mew::destroy_buffer(device, &mut allocator, &mut self.vertex_buffer);
            mew::destroy_buffer(device, &mut allocator, &mut self.index_buffer);
            mew::destroy_buffer(device, &mut allocator, &mut self.object_buffer);
            mew::destroy_buffer(device, &mut allocator, &mut self.mesh_buffer);
            mew::destroy_buffer(device, &mut allocator, &mut self.draw_indirect_buffer);
            mew::destroy_buffer(device, &mut allocator, &mut self.dispatch_buffer);

            device.destroy_pipeline(self.copy_pipeline, None);
            device.destroy_pipeline(self.draw_pipeline, None);
            device.destroy_pipeline(self.cull_pipeline, None);
        }
    }
}

struct App {
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(_) = self.renderer.as_ref() {
            return;
        }

        let window_width: u32 = 1700;
        let window_height: u32 = 900;

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("window: mew-rust")
                    .with_resizable(true)
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        f64::from(window_width),
                        f64::from(window_height),
                    )),
            )
            .expect("Window creation failed");

        self.renderer = Some(Renderer::new(window));
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer
                        .camera
                        .process_mouse_input(delta.0 as f32, delta.1 as f32);
                }
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // This should be providing us with new dims, but it's returning outdated dims which can cause OOB surface extent - possibly Niri specific?
            winit::event::WindowEvent::Resized(_) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    recreate_swapchain(&mut renderer.backend, &renderer.window);
                    recreate_resources_on_swapchain_resize(renderer);

                    let window_size = renderer.window.inner_size();
                    println!(
                        "Swapchain resize: {}x{}",
                        window_size.width, window_size.height
                    );
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(renderer) = self.renderer.as_mut() {
                    render_loop(renderer);

                    if renderer.backend.swapchain.dirty {
                        recreate_swapchain(&mut renderer.backend, &renderer.window);
                        recreate_resources_on_swapchain_resize(renderer);

                        let window_size = renderer.window.inner_size();
                        println!(
                            "Swapchain resize: {}x{}",
                            window_size.width, window_size.height
                        );
                    }
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: Code(code),
                        state: element_state,
                        ..
                    },
                ..
            } => {
                let camera = &mut self.renderer.as_mut().unwrap().camera;

                match (code, element_state) {
                    (KeyCode::Escape, ElementState::Pressed) => {
                        event_loop.exit();
                    }
                    (KeyCode::KeyW, ElementState::Pressed) => {
                        camera.process_input(Key::W, KeyState::Pressed)
                    }
                    (KeyCode::KeyS, ElementState::Pressed) => {
                        camera.process_input(Key::S, KeyState::Pressed)
                    }
                    (KeyCode::KeyA, ElementState::Pressed) => {
                        camera.process_input(Key::A, KeyState::Pressed)
                    }
                    (KeyCode::KeyD, ElementState::Pressed) => {
                        camera.process_input(Key::D, KeyState::Pressed)
                    }
                    (KeyCode::KeyW, ElementState::Released) => {
                        camera.process_input(Key::W, KeyState::Released)
                    }
                    (KeyCode::KeyS, ElementState::Released) => {
                        camera.process_input(Key::S, KeyState::Released)
                    }
                    (KeyCode::KeyA, ElementState::Released) => {
                        camera.process_input(Key::A, KeyState::Released)
                    }
                    (KeyCode::KeyD, ElementState::Released) => {
                        camera.process_input(Key::D, KeyState::Released)
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_mut() {
            let start = Instant::now();
            if let Some(last_frame_time) = renderer.last_frame_time {
                let delta_time = (start - last_frame_time).as_secs_f32();
                renderer.last_frame_time = Some(start);
                renderer.camera.update(delta_time);
            } else {
                renderer.last_frame_time = Some(start);
            }

            renderer.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_ref() {
            unsafe { renderer.backend.device.device_wait_idle().unwrap() };
        }

        self.renderer = None;
    }
}

fn render_loop(renderer: &mut Renderer) {
    let device = &renderer.backend.device;
    let current_index = renderer.frame_index % mew::FRAMES_IN_FLIGHT;

    let aspect = (renderer.backend.swapchain.extent.width as f32)
        / (renderer.backend.swapchain.extent.height as f32);
    let view = renderer.camera.get_view_matrix();
    let proj = mew::get_infinite_reverse_perspective_matrix(
        renderer.camera.fovy,
        aspect as f32,
        renderer.camera.near,
    );
    let view_proj = proj * view;

    let frame_resource = &renderer.backend.frame_resources[current_index];
    let fence = frame_resource.fence;
    unsafe {
        device.wait_for_fences(&[fence], true, 1000000000).unwrap();
    }

    let acquire_semaphore = frame_resource.image_acquired_semaphore;

    let swapchain_idx: usize;
    unsafe {
        let acquire_result = renderer.backend.swapchain.loader.acquire_next_image(
            renderer.backend.swapchain.swapchain,
            1000000000,
            acquire_semaphore,
            Fence::null(),
        );

        match acquire_result {
            Ok((present_idx, _)) => {
                swapchain_idx = present_idx as usize;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                renderer.backend.swapchain.dirty = true;
                return;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {e:?}");
            }
        }
        device.reset_fences(&[fence]).unwrap();
    }

    unsafe {
        device
            .reset_command_pool(
                frame_resource.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )
            .unwrap();
    }

    let cmd = frame_resource.command_buffer;
    let cmd_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(cmd, &cmd_begin_info).unwrap() };

    unsafe {
        renderer
            .backend
            .descriptor
            .sets
            .iter()
            .enumerate()
            .for_each(|(index, descriptor)| {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    renderer.backend.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    renderer.backend.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
            });
    }

    if renderer.frame_index >= mew::FRAMES_IN_FLIGHT {
        let mut pipeline_query_results = [0u64; CURRENT_QUERIES as usize];
        unsafe {
            device
                .get_query_pool_results(
                    frame_resource.pipeline_query,
                    0,
                    &mut pipeline_query_results,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap();
        }

        // TODO: egui?
        let _triangle_count = pipeline_query_results[0];
        // dbg!(_triangle_count);
    }

    unsafe {
        device.reset_query_pool(frame_resource.pipeline_query, 0, CURRENT_QUERIES);
    }

    let mut image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .image(renderer.backend.swapchain.images[swapchain_idx])
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
    // draw image & depth image
    image_memory_barrier.image = renderer.framebuffer.draw_image.image;
    image_memory_barrier.new_layout = vk::ImageLayout::GENERAL;
    image_memory_barriers.push(image_memory_barrier);
    image_memory_barrier.image = renderer.framebuffer.depth_image.image;
    image_memory_barrier.new_layout = vk::ImageLayout::GENERAL;
    image_memory_barrier.subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    image_memory_barriers.push(image_memory_barrier);

    let mut dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

    let renderables_count =
        renderer.object_buffer.size as usize / std::mem::size_of::<mew::loader::ObjectData>();

    // Pass 0 - Zero buffers
    unsafe {
        device.cmd_fill_buffer(cmd, renderer.dispatch_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
    }
    giga_barrier(device, cmd);

    // Pass 1 - Culling
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, renderer.cull_pipeline);

        let proj_t = proj.transpose();
        let m0 = proj_t.x_axis;
        let m1 = proj_t.y_axis;
        let m3 = proj_t.w_axis;
        let left_plane = (m3 + m0).xyz().normalize();
        let bottom_plane = (m3 + m1).xyz().normalize();

        let planes = glam::Vec4::new(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
        let p00 = proj.x_axis.x;
        let p11 = proj.y_axis.y;

        let cull_constants = basic_renderer::CullConstants {
            view,
            mesh_buffer: renderer.mesh_buffer.address,
            object_buffer: renderer.object_buffer.address,
            draw_indirect_buffer: renderer.draw_indirect_buffer.address,
            dispatch_buffer: renderer.dispatch_buffer.address,
            planes,
            p00,
            p11,
            near: renderer.camera.near,
            far: renderer.camera.far,
            count: renderables_count as u32,
        };

        device.cmd_push_constants(cmd, renderer.backend.pipeline_layout, vk::ShaderStageFlags::ALL, 0, mew::push_constants_as_bytes(&cull_constants));
        let group_count_x = mew::get_group_count(renderables_count as u32, 256);
        device.cmd_dispatch(cmd, group_count_x, 1, 1);
    };

    mew::giga_barrier(device, cmd);

    // Pass 2 - Rasterization
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, renderer.draw_pipeline);

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        };

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(renderer.framebuffer.draw_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value)];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(renderer.framebuffer.depth_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value);

        let swapchain_extent = renderer.backend.swapchain.extent;

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                extent: swapchain_extent,
                ..Default::default()
            })
            .depth_attachment(&depth_attachment)
            .color_attachments(&color_attachments)
            .layer_count(1);

        device.cmd_begin_rendering(cmd, &rendering_info);

        let viewport = [vk::Viewport::default()
            .y(swapchain_extent.height as f32)
            .width(swapchain_extent.width as f32)
            .height(-(swapchain_extent.height as f32))
            .max_depth(1.0)
            .min_depth(0.0)];

        let scissors = [vk::Rect2D {
            extent: swapchain_extent,
            ..Default::default()
        }];

        device.cmd_set_viewport(cmd, 0, &viewport);
        device.cmd_set_scissor(cmd, 0, &scissors);

        let mesh_constants = basic_renderer::MeshConstants {
            view_proj,
            vertex_buffer: renderer.vertex_buffer.address,
            mesh_buffer: renderer.mesh_buffer.address,
            object_buffer: renderer.object_buffer.address,
        };

        device.cmd_push_constants(cmd, renderer.backend.pipeline_layout, vk::ShaderStageFlags::ALL, 0, mew::push_constants_as_bytes(&mesh_constants));
        device.cmd_bind_index_buffer(cmd, renderer.index_buffer.buffer, 0, vk::IndexType::UINT32);

        device.cmd_begin_query(
            cmd,
            frame_resource.pipeline_query,
            0,
            vk::QueryControlFlags::empty(),
        );

        device.cmd_draw_indexed_indirect_count(
            cmd,
            renderer.draw_indirect_buffer.buffer,
            0,
            renderer.dispatch_buffer.buffer,
            0,
            renderables_count as u32, // There is a physical limit but for non-meshlets we are not concerned
            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
        );
        device.cmd_end_query(cmd, frame_resource.pipeline_query, 0);

        device.cmd_end_rendering(cmd);
    };

    mew::giga_barrier(&device, cmd);

    // Copy to swapchain
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, renderer.copy_pipeline);
        let copy_constants = basic_renderer::CopySwapchainConstants {
            src_id: renderer.framebuffer.draw_index,
            dst_id: renderer.framebuffer.swapchain_indices[swapchain_idx],
        };
        device.cmd_push_constants(cmd, renderer.backend.pipeline_layout, vk::ShaderStageFlags::ALL, 0, mew::push_constants_as_bytes(&copy_constants));
        let group_count_x = mew::get_group_count(renderer.backend.swapchain.extent.width, 8);
        let group_count_y = mew::get_group_count(renderer.backend.swapchain.extent.height, 8);
        device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    }

    // transition to present
    image_memory_barrier.image = renderer.backend.swapchain.images[swapchain_idx];
    image_memory_barrier.old_layout = vk::ImageLayout::GENERAL;
    image_memory_barrier.new_layout = vk::ImageLayout::PRESENT_SRC_KHR;
    image_memory_barrier.subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
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

    let render_semaphore = renderer.backend.render_done_semaphores[swapchain_idx];
    let semaphore_signal_info = vk::SemaphoreSubmitInfo::default()
        .semaphore(render_semaphore)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);

    let render_semaphore = [render_semaphore];
    let semaphore_signal_info = [semaphore_signal_info];
    let semaphore_wait_info = [semaphore_wait_info];
    let cmd_submit_info = [cmd_submit_info];
    let swapchain_idx = [swapchain_idx as u32];
    let swapchain = [renderer.backend.swapchain.swapchain];

    let submit_info = vk::SubmitInfo2::default()
        .signal_semaphore_infos(&semaphore_signal_info)
        .wait_semaphore_infos(&semaphore_wait_info)
        .command_buffer_infos(&cmd_submit_info);

    unsafe {
        device
            .queue_submit2(renderer.backend.graphics_queue, &[submit_info], fence)
            .unwrap();
    }

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&render_semaphore)
        .image_indices(&swapchain_idx)
        .swapchains(&swapchain);

    unsafe {
        let present_result = renderer
            .backend
            .swapchain
            .loader
            .queue_present(renderer.backend.graphics_queue, &present_info);

        match present_result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                renderer.backend.swapchain.dirty = true;
                return;
            }
            Err(e) => {
                panic!("Present error: {e:?}");
            }
        }
    };

    renderer.frame_index += 1;
}

// This is project specific
fn recreate_resources_on_swapchain_resize(renderer: &mut Renderer) {
    let device = &renderer.backend.device;

    let mut allocator = renderer.backend.allocator.lock().unwrap();

    // Destroy outdated resources
    mew::destroy_image(device, &mut allocator, &mut renderer.framebuffer.draw_image);
    mew::destroy_image(
        device,
        &mut allocator,
        &mut renderer.framebuffer.depth_image,
    );

    // Recreate resources
    renderer.framebuffer.draw_image = mew::create_image(
        &device,
        &mut allocator,
        renderer.backend.swapchain.extent,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::ImageAspectFlags::COLOR,
        false,
    );

    renderer.framebuffer.depth_image = mew::create_image(
        &device,
        &mut allocator,
        renderer.backend.swapchain.extent,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        false,
    );

    // Update descriptors
    renderer.backend.update_image_descriptor(
        renderer.framebuffer.draw_index,
        renderer.framebuffer.draw_image.view,
        RenderResourceTag::Storage,
    );

    for (index, view) in renderer
        .framebuffer
        .swapchain_indices
        .iter()
        .zip(renderer.backend.swapchain.views.iter())
    {
        renderer
            .backend
            .update_image_descriptor(*index, *view, RenderResourceTag::Storage);
    }
}

fn main() {
    // unsafe {
    //     std::env::remove_var("WAYLAND_DISPLAY");
    // }

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { renderer: None }).unwrap();
}
