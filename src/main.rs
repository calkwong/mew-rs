use ash::vk::{self, Fence, Semaphore};
use glam::Vec4Swizzles;
use mew::descriptors::RenderResourceTag;
use mew::swapchain::recreate_swapchain;
use mew::{
    FrameData,
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

const MAX_QUERY_COUNT: u32 = 10;
const CURRENT_QUERIES: u32 = 1;
const MAX_PUSH_CONSTANTS_SIZE: u32 = 256;

struct State {
    window: Window,
    camera: Camera,
    last_frame_time: Option<Instant>,
    engine: mew::Engine,
    images: Images,
    frame_data: [mew::FrameData; mew::FRAMES_IN_FLIGHT],
    render_done_semaphores: Vec<Semaphore>,
    frame_index: usize,
    copy_pipeline: vk::Pipeline,
    draw_pipeline: vk::Pipeline,
    cull_pipeline: vk::Pipeline,
    vertex_buffer: mew::Buffer,
    index_buffer: mew::Buffer,
    mesh_buffer: mew::Buffer,
    object_buffer: mew::Buffer,
    draw_indirect_buffer: mew::Buffer,
    dispatch_buffer: mew::Buffer,
}

struct Images {
    draw_image: mew::Image,
    depth_image: mew::Image,
    draw_index: u32,
    swapchain_indices: Vec<u32>,
}

impl State {
    fn new(window: Option<Window>) -> Self {
        let engine = mew::Engine::new(window.as_ref());

        assert_eq!(
            engine.properties.limits.max_push_constants_size,
            MAX_PUSH_CONSTANTS_SIZE
        );

        let camera = Camera::default().position(glam::Vec3::new(0.0, 0.0, 5.0));

        let device = &engine.device;

        let allocator = (*engine.allocator).clone();
        let mut allocator = &mut allocator.lock().unwrap();

        let draw_image = mew::create_image(
            &device,
            &mut allocator,
            engine.swapchain.extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let depth_image = mew::create_image(
            &device,
            &mut allocator,
            engine.swapchain.extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            false,
        );

        let draw_index = engine.register_image(draw_image.view, RenderResourceTag::Storage);

        let swapchain_indices: Vec<u32> = engine
            .swapchain
            .views
            .iter()
            .map(|view| engine.register_image(*view, RenderResourceTag::Storage))
            .collect();

        // Prepare frame data
        let mut frame_data: [FrameData; mew::FRAMES_IN_FLIGHT] =
            [FrameData::default(); mew::FRAMES_IN_FLIGHT];

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(engine.queue_family_index);

        unsafe {
            (0..mew::FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].command_pool = device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap();
            });
        }

        unsafe {
            (0..mew::FRAMES_IN_FLIGHT).for_each(|i| {
                let pool = &frame_data[i].command_pool;

                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                frame_data[i].command_buffer = device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()[0];
            });
        }

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            (0..mew::FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].fence = device.create_fence(&fence_info, None).unwrap();
            });
        }

        let semaphore_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            (0..mew::FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].image_acquired_semaphore =
                    device.create_semaphore(&semaphore_info, None).unwrap()
            });
        }

        let render_done_semaphores: Vec<vk::Semaphore> = unsafe {
            (0..engine.swapchain.images.len())
                .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
                .collect()
        };

        let query_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .query_count(MAX_QUERY_COUNT)
            .pipeline_statistics(vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS);

        unsafe {
            (0..mew::FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].pipeline_query = device.create_query_pool(&query_info, None).unwrap();
            });
        }

        let copy_shader = mew::load_shader(device, "shaders/compiled/copy_swapchain.spv");
        let draw_shader = mew::load_shader(device, "shaders/compiled/mesh.spv");
        let cull_shader = mew::load_shader(device, "shaders/compiled/culling.spv");

        let cull_pipeline =
            mew::create_compute_pipeline(&device, engine.pipeline_layout, cull_shader);
        let copy_pipeline =
            mew::create_compute_pipeline(&device, engine.pipeline_layout, copy_shader);
        let draw_pipeline = mew::create_graphics_pipeline(
            &device,
            engine.pipeline_layout,
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

        let (vertices, len) = mew::as_bytes(&scene.vertices);
        let vertex_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            len,
            vertices,
        );

        let (indices, len) = mew::as_bytes(&scene.indices);
        let index_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDEX_BUFFER,
            len,
            indices,
        );

        let (meshes, len) = mew::as_bytes(&scene.meshes);
        let mesh_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            len,
            meshes,
        );

        let (objects, len) = mew::as_bytes(&scene.renderables);
        let object_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            len,
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
            window: window.unwrap(),
            camera,
            last_frame_time: None,
            engine,
            images: Images {
                draw_image,
                depth_image,
                draw_index,
                swapchain_indices,
            },
            frame_data,
            render_done_semaphores,
            frame_index: 0,
            copy_pipeline,
            draw_pipeline,
            cull_pipeline,
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

impl Drop for State {
    fn drop(&mut self) {
        let device = &self.engine.device;

        unsafe {
            self.frame_data.iter().for_each(|data| {
                device.destroy_command_pool(data.command_pool, None);
                device.destroy_fence(data.fence, None);
                device.destroy_semaphore(data.image_acquired_semaphore, None);
                device.destroy_query_pool(data.pipeline_query, None);
            });

            self.render_done_semaphores.iter().for_each(|semaphore| {
                device.destroy_semaphore(*semaphore, None);
            });

            let mut allocator = self.engine.allocator.lock().unwrap();

            // Destroy resources
            mew::destroy_image(device, &mut allocator, &mut self.images.draw_image);
            mew::destroy_image(device, &mut allocator, &mut self.images.depth_image);
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                if let Some(state) = self.state.as_mut() {
                    state
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
                if let Some(state) = self.state.as_mut() {
                    recreate_swapchain(&mut state.engine, &state.window);
                    recreate_resources_on_swapchain_resize(state);

                    let window_size = state.window.inner_size();
                    println!(
                        "Swapchain resize: {}x{}",
                        window_size.width, window_size.height
                    );
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    render_loop(state);

                    if state.engine.swapchain.dirty {
                        recreate_swapchain(&mut state.engine, &state.window);
                        recreate_resources_on_swapchain_resize(state);

                        let window_size = state.window.inner_size();
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
                let camera = &mut self.state.as_mut().unwrap().camera;

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
        if let Some(state) = self.state.as_mut() {
            let start = Instant::now();
            if let Some(last_frame_time) = state.last_frame_time {
                let delta_time = (start - last_frame_time).as_secs_f32();
                state.last_frame_time = Some(start);
                state.camera.update(delta_time);
            } else {
                state.last_frame_time = Some(start);
            }

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
    let current_index = state.frame_index % mew::FRAMES_IN_FLIGHT;

    let aspect = (state.engine.swapchain.extent.width as f32)
        / (state.engine.swapchain.extent.height as f32);
    let view = state.camera.get_view_matrix();
    let proj = mew::get_infinite_reverse_perspective_matrix(
        state.camera.fovy,
        aspect as f32,
        state.camera.near,
    );
    let view_proj = proj * view;

    let frame_data = &state.frame_data[current_index];
    let fence = frame_data.fence;
    unsafe {
        device.wait_for_fences(&[fence], true, 1000000000).unwrap();
    }

    let acquire_semaphore = frame_data.image_acquired_semaphore;

    let swapchain_idx: usize;
    unsafe {
        let acquire_result = state.engine.swapchain.loader.acquire_next_image(
            state.engine.swapchain.swapchain,
            1000000000,
            acquire_semaphore,
            Fence::null(),
        );

        match acquire_result {
            Ok((present_idx, _)) => {
                swapchain_idx = present_idx as usize;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                state.engine.swapchain.dirty = true;
                return;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {e:?}");
            }
        }
        device.reset_fences(&[fence]).unwrap();
    }

    let mut push_constants_scratch = [0u8; MAX_PUSH_CONSTANTS_SIZE as usize];

    unsafe {
        device
            .reset_command_pool(frame_data.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
    }

    let cmd = frame_data.command_buffer;
    let cmd_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(cmd, &cmd_begin_info).unwrap() };

    unsafe {
        state
            .engine
            .descriptor
            .sets
            .iter()
            .enumerate()
            .for_each(|(index, descriptor)| {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    state.engine.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    state.engine.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
            });
    }

    if state.frame_index >= mew::FRAMES_IN_FLIGHT {
        let mut pipeline_query_results = [0u64; CURRENT_QUERIES as usize];
        unsafe {
            device
                .get_query_pool_results(
                    frame_data.pipeline_query,
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
        device.reset_query_pool(frame_data.pipeline_query, 0, MAX_QUERY_COUNT);
    }

    let mut image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .image(state.engine.swapchain.images[swapchain_idx])
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
    image_memory_barrier.image = state.images.draw_image.image;
    image_memory_barrier.new_layout = vk::ImageLayout::GENERAL;
    image_memory_barriers.push(image_memory_barrier);
    image_memory_barrier.image = state.images.depth_image.image;
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
        state.object_buffer.size as usize / std::mem::size_of::<mew::loader::ObjectData>();

    // Pass 0 - Zero buffers
    unsafe {
        device.cmd_fill_buffer(cmd, state.dispatch_buffer.buffer, 0, vk::WHOLE_SIZE, 0);
    }
    giga_barrier(device, cmd);

    // Pass 1 - Culling
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.cull_pipeline);

        let proj_t = proj.transpose();
        let m0 = proj_t.x_axis;
        let m1 = proj_t.y_axis;
        let m3 = proj_t.w_axis;
        let left_plane = (m3 + m0).xyz().normalize();
        let bottom_plane = (m3 + m1).xyz().normalize();

        let planes = glam::Vec4::new(left_plane.x, left_plane.z, bottom_plane.y, bottom_plane.z);
        let p00 = proj.x_axis.x;
        let p11 = proj.y_axis.y;

        #[repr(C)]
        struct PushConstants {
            view: glam::Mat4,
            mesh_buffer: vk::DeviceAddress,
            object_buffer: vk::DeviceAddress,
            draw_indirect_buffer: vk::DeviceAddress,
            dispatch_buffer: vk::DeviceAddress,
            planes: glam::Vec4,
            p00: f32,
            p11: f32,
            near: f32,
            far: f32,
            count: u32,
        }
        let pc = PushConstants {
            view,
            mesh_buffer: state.mesh_buffer.address,
            object_buffer: state.object_buffer.address,
            draw_indirect_buffer: state.draw_indirect_buffer.address,
            dispatch_buffer: state.dispatch_buffer.address,
            planes,
            p00,
            p11,
            near: state.camera.near,
            far: state.camera.far,
            count: renderables_count as u32,
        };
        mew::push_constants(
            &device,
            cmd,
            state.engine.pipeline_layout,
            &pc,
            &mut push_constants_scratch,
        );

        let group_count_x = mew::get_group_count(renderables_count as u32, 256);
        device.cmd_dispatch(cmd, group_count_x, 1, 1);
    };

    mew::giga_barrier(device, cmd);

    // Pass 2 - Rasterization
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, state.draw_pipeline);

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
            .image_view(state.images.draw_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value)];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(state.images.depth_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value);

        let swapchain_extent = state.engine.swapchain.extent;

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

        #[allow(dead_code)]
        struct PushConstants {
            view_proj: glam::Mat4,
            vertex_buffer: vk::DeviceAddress,
            mesh_buffer: vk::DeviceAddress,
            object_buffer: vk::DeviceAddress,
        }
        let pc = PushConstants {
            view_proj,
            vertex_buffer: state.vertex_buffer.address,
            mesh_buffer: state.mesh_buffer.address,
            object_buffer: state.object_buffer.address,
        };
        mew::push_constants(
            &device,
            cmd,
            state.engine.pipeline_layout,
            &pc,
            &mut push_constants_scratch,
        );

        device.cmd_bind_index_buffer(cmd, state.index_buffer.buffer, 0, vk::IndexType::UINT32);

        device.cmd_begin_query(
            cmd,
            frame_data.pipeline_query,
            0,
            vk::QueryControlFlags::empty(),
        );

        device.cmd_draw_indexed_indirect_count(
            cmd,
            state.draw_indirect_buffer.buffer,
            0,
            state.dispatch_buffer.buffer,
            0,
            renderables_count as u32, // There is a physical limit but for non-meshlets we are not concerned
            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
        );
        device.cmd_end_query(cmd, frame_data.pipeline_query, 0);

        device.cmd_end_rendering(cmd);
    };

    mew::giga_barrier(&device, cmd);

    // Copy to swapchain
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.copy_pipeline);
        #[allow(dead_code)]
        struct PushConstants {
            src_id: u32,
            dst_id: u32,
        }
        let pc = PushConstants {
            src_id: state.images.draw_index,
            dst_id: state.images.swapchain_indices[swapchain_idx],
        };
        mew::push_constants(
            &device,
            cmd,
            state.engine.pipeline_layout,
            &pc,
            &mut push_constants_scratch,
        );
        let group_count_x = mew::get_group_count(state.engine.swapchain.extent.width, 8);
        let group_count_y = mew::get_group_count(state.engine.swapchain.extent.height, 8);
        device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    }

    // transition to present
    image_memory_barrier.image = state.engine.swapchain.images[swapchain_idx];
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
    let swapchain = [state.engine.swapchain.swapchain];

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
            .swapchain
            .loader
            .queue_present(state.engine.graphics_queue, &present_info);

        match present_result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                state.engine.swapchain.dirty = true;
                return;
            }
            Err(e) => {
                panic!("Present error: {e:?}");
            }
        }
    };

    state.frame_index += 1;
}

// This is project specific
fn recreate_resources_on_swapchain_resize(state: &mut State) {
    let device = &state.engine.device;

    let mut allocator = state.engine.allocator.lock().unwrap();

    // Destroy outdated resources
    mew::destroy_image(device, &mut allocator, &mut state.images.draw_image);
    mew::destroy_image(device, &mut allocator, &mut state.images.depth_image);

    // Recreate resources
    state.images.draw_image = mew::create_image(
        &device,
        &mut allocator,
        state.engine.swapchain.extent,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::ImageAspectFlags::COLOR,
        false,
    );

    state.images.depth_image = mew::create_image(
        &device,
        &mut allocator,
        state.engine.swapchain.extent,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        false,
    );

    // Update descriptors
    state.engine.update_image_descriptor(
        state.images.draw_index,
        state.images.draw_image.view,
        RenderResourceTag::Storage,
    );

    for (index, view) in state
        .images
        .swapchain_indices
        .iter()
        .zip(state.engine.swapchain.views.iter())
    {
        state
            .engine
            .update_image_descriptor(*index, *view, RenderResourceTag::Storage);
    }
}

fn main() {
    // unsafe {
    //     std::env::remove_var("WAYLAND_DISPLAY");
    // }

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { state: None }).unwrap();
}
