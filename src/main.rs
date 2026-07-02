use ash::vk::{self, Fence, Semaphore};
use mew::{
    FrameData,
    camera::{Camera, Key, KeyState},
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey::Code},
    window::{Window, WindowId},
};

const FRAMES_IN_FLIGHT: usize = 2;
const MAX_QUERY_COUNT: u32 = 10;
const CURRENT_QUERIES: u32 = 1;

struct State {
    window: Window,
    camera: Camera,
    last_frame_time: Option<Instant>,
    engine: mew::Engine,
    draw_image: mew::Image,
    depth_image: mew::Image,
    frame_data: [mew::FrameData; FRAMES_IN_FLIGHT],
    render_done_semaphores: Vec<Semaphore>,
    frame_index: usize,
    copy_pipeline: vk::Pipeline,
    draw_pipeline: vk::Pipeline,
    cull_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: [vk::DescriptorSet; 4],
    vertex_buffer: mew::Buffer,
    index_buffer: mew::Buffer,
    mesh_buffer: mew::Buffer,
    object_buffer: mew::Buffer,
    draw_indirect_buffer: mew::Buffer,
}

impl State {
    fn new(window: Option<Window>) -> Self {
        let mut engine = mew::Engine::new(window.as_ref());
        let camera = Camera::default().position(glam::Vec3::new(0.0, 0.0, 5.0));

        let device = &engine.device;

        let draw_image = mew::create_image(
            &device,
            &mut engine.allocator,
            engine.swapchain.extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            false,
        );

        let depth_image = mew::create_image(
            &device,
            &mut engine.allocator,
            engine.swapchain.extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            false,
        );

        // frame data
        let mut frame_data: [FrameData; FRAMES_IN_FLIGHT] =
            [FrameData::default(); FRAMES_IN_FLIGHT];

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(engine.queue_family_index);

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].command_pool = device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap();
            });
        }

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                let pool = &frame_data[i].command_pool;

                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                // note: a single command buffer per command pool
                frame_data[i].command_buffer = device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()[0];
            });
        }

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].fence = device.create_fence(&fence_info, None).unwrap();
            });
        }

        let semaphore_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
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
            (0..FRAMES_IN_FLIGHT).for_each(|i| {
                frame_data[i].pipeline_query = device.create_query_pool(&query_info, None).unwrap();
            });
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
        let descriptor_pool = mew::descriptors::create_descriptor_pool(&device, &pool_size);
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
            mew::descriptors::create_descriptor_layouts(&device, buffer_binding, &[]);
        let storage_descriptor_layout = mew::descriptors::create_descriptor_layouts(
            &device,
            storage_image_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );
        let sample_descriptor_layout = mew::descriptors::create_descriptor_layouts(
            &device,
            sample_image_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );
        let sampler_descriptor_layout = mew::descriptors::create_descriptor_layouts(
            &device,
            sampler_binding,
            &[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        );

        let buffer_descriptor = mew::descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            buffer_descriptor_layout,
            3,
        );
        let storage_descriptor = mew::descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            storage_descriptor_layout,
            300,
        );
        let sample_descriptor = mew::descriptors::create_descriptor_sets(
            &device,
            descriptor_pool,
            sample_descriptor_layout,
            300,
        );
        let sampler_descriptor = mew::descriptors::create_descriptor_sets(
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

        // TODO: create fn for write desc set
        let mut image_infos: Vec<vk::DescriptorImageInfo> =
            Vec::from([vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(draw_image.view)]);
        engine.swapchain.views.iter().for_each(|image_view| {
            image_infos.push(
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(*image_view),
            );
        });

        // Write to descriptors
        let storage_descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(storage_descriptor)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos)
            .descriptor_count(4);
        unsafe {
            device.update_descriptor_sets(&[storage_descriptor_write], &[]);
        }

        let descriptor_set_layouts = [
            buffer_descriptor_layout,
            storage_descriptor_layout,
            sample_descriptor_layout,
            sampler_descriptor_layout,
        ];

        let push_constant_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(engine.properties.limits.max_push_constants_size)];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_range);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        // Clones device and moves handles no longer needed to deletion stack
        // These handles are no longer valid in this scope after move
        for layout in descriptor_set_layouts {
            let device_clone = device.clone();
            engine.deletion_stack.push(move || unsafe {
                device_clone.destroy_descriptor_set_layout(layout, None);
            });
        }
        let device_clone = device.clone();
        engine.deletion_stack.push(move || unsafe {
            device_clone.destroy_descriptor_pool(descriptor_pool, None);
        });

        let copy_shader = mew::load_shader(device, "shaders/compiled/copy_swapchain.spv");
        let draw_shader = mew::load_shader(device, "shaders/compiled/mesh.spv");
        let cull_shader = mew::load_shader(device, "shaders/compiled/culling.spv");

        let cull_pipeline = mew::create_compute_pipeline(&device, pipeline_layout, cull_shader);
        let copy_pipeline = mew::create_compute_pipeline(&device, pipeline_layout, copy_shader);
        let draw_pipeline = mew::create_graphics_pipeline(
            &device,
            pipeline_layout,
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

        let vertex_size = scene.vertices.len() * std::mem::size_of::<mew::loader::Vertex>();
        let vertices = unsafe {
            std::slice::from_raw_parts(scene.vertices.as_ptr() as *const u8, vertex_size)
        };

        let vertex_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].fence,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut engine.allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vertex_size as u64,
            vertices,
        );

        let indices_size = scene.indices.len() * std::mem::size_of::<u32>();
        let indices = unsafe {
            std::slice::from_raw_parts(scene.indices.as_ptr() as *const u8, indices_size)
        };
        let index_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].fence,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut engine.allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDEX_BUFFER,
            indices_size as u64,
            indices,
        );

        let meshes_size = scene.meshes.len() * std::mem::size_of::<mew::loader::Mesh>();
        let meshes =
            unsafe { std::slice::from_raw_parts(scene.meshes.as_ptr() as *const u8, meshes_size) };
        let mesh_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].fence,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut engine.allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            meshes_size as u64,
            meshes,
        );

        let object_size = scene.renderables.len() * std::mem::size_of::<mew::loader::ObjectData>();
        let objects = unsafe {
            std::slice::from_raw_parts(scene.renderables.as_ptr() as *const u8, object_size)
        };
        let object_buffer = mew::create_buffer_with_data(
            device,
            engine.graphics_queue,
            frame_data[0].fence,
            frame_data[0].command_pool,
            frame_data[0].command_buffer,
            &mut engine.allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            object_size as u64,
            objects,
        );

        let draw_indirect_buffer = mew::create_buffer(
            device,
            &mut engine.allocator,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            (scene.renderables.len() * std::mem::size_of::<vk::DrawIndexedIndirectCommand>())
                as u64,
        );

        draw_indirect_buffer.size;

        let this = Self {
            window: window.unwrap(),
            camera,
            last_frame_time: None,
            engine,
            draw_image,
            depth_image,
            frame_data,
            render_done_semaphores,
            frame_index: 0,
            copy_pipeline,
            draw_pipeline,
            cull_pipeline,
            pipeline_layout,
            descriptor_sets,
            vertex_buffer,
            index_buffer,
            mesh_buffer,
            object_buffer,
            draw_indirect_buffer,
        };

        this
    }
}

impl Drop for State {
    fn drop(&mut self) {
        let device = &self.engine.device;

        unsafe {
            // TODO: drop?
            self.frame_data.iter().for_each(|data| {
                device.destroy_command_pool(data.command_pool, None);
                device.destroy_fence(data.fence, None);
                device.destroy_semaphore(data.image_acquired_semaphore, None);
                device.destroy_query_pool(data.pipeline_query, None);
            });

            self.render_done_semaphores.iter().for_each(|semaphore| {
                device.destroy_semaphore(*semaphore, None);
            });

            // clean up allocator + resources
            // TODO: consider drop or ManuallyDrop image?
            mew::destroy_image(device, &mut self.engine.allocator, &mut self.draw_image);
            mew::destroy_image(device, &mut self.engine.allocator, &mut self.depth_image);
            mew::destroy_buffer(device, &mut self.engine.allocator, &mut self.vertex_buffer);
            mew::destroy_buffer(device, &mut self.engine.allocator, &mut self.index_buffer);
            mew::destroy_buffer(device, &mut self.engine.allocator, &mut self.object_buffer);
            mew::destroy_buffer(device, &mut self.engine.allocator, &mut self.mesh_buffer);
            mew::destroy_buffer(
                device,
                &mut self.engine.allocator,
                &mut self.draw_indirect_buffer,
            );

            device.destroy_pipeline_layout(self.pipeline_layout, None);
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
                    let window_size = state.window.inner_size();
                    let new_extent = vk::Extent2D {
                        width: window_size.width,
                        height: window_size.height,
                    };

                    mew::recreate_swapchain(&mut state.engine, new_extent);
                    recreate_resources_on_swapchain_resize(state);

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
                        let window_size = state.window.inner_size();

                        let new_extent = vk::Extent2D {
                            width: window_size.width as u32,
                            height: window_size.height as u32,
                        };

                        mew::recreate_swapchain(&mut state.engine, new_extent);
                        recreate_resources_on_swapchain_resize(state);
                        state.engine.swapchain.dirty = false;
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
    let current_index = state.frame_index % FRAMES_IN_FLIGHT;

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
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();

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
            .descriptor_sets
            .iter()
            .enumerate()
            .for_each(|(index, descriptor)| {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    state.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    state.pipeline_layout,
                    index as u32,
                    &[*descriptor],
                    &[],
                );
            });
    }

    let acquire_semaphore = frame_data.image_acquired_semaphore;

    let swapchain_idx: usize;
    unsafe {
        let acquire_result = state.engine.swapchain.loader.acquire_next_image(
            state.engine.swapchain.swapchain,
            u64::MAX,
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

    if state.frame_index >= FRAMES_IN_FLIGHT {
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

        let triangle_count = pipeline_query_results[0];
        dbg!(triangle_count);
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
    image_memory_barrier.image = state.draw_image.image;
    image_memory_barrier.new_layout = vk::ImageLayout::GENERAL;
    image_memory_barriers.push(image_memory_barrier);
    image_memory_barrier.image = state.depth_image.image;
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

    // hack
    let renderables_count =
        state.object_buffer.size as usize / std::mem::size_of::<mew::loader::ObjectData>();

    // dumb compute shader that builds draw indirect buffer
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.cull_pipeline);

        #[allow(dead_code)]
        struct PushConstants {
            mesh_buffer: vk::DeviceAddress,
            object_buffer: vk::DeviceAddress,
            draw_indirect_buffer: vk::DeviceAddress,
            count: u32,
        }
        let pc = PushConstants {
            mesh_buffer: state.mesh_buffer.address,
            object_buffer: state.object_buffer.address,
            draw_indirect_buffer: state.draw_indirect_buffer.address,
            count: renderables_count as u32,
        };
        mew::push_constants(&device, cmd, state.pipeline_layout, &pc);

        let group_count_x = mew::get_group_count(renderables_count as u32, 256);
        device.cmd_dispatch(cmd, group_count_x, 1, 1);
    };

    mew::giga_barrier(device, cmd);

    // rasterize
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
            .image_view(state.draw_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value)];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(state.depth_image.view)
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
        mew::push_constants(&device, cmd, state.pipeline_layout, &pc);

        device.cmd_bind_index_buffer(cmd, state.index_buffer.buffer, 0, vk::IndexType::UINT32);

        device.cmd_begin_query(
            cmd,
            frame_data.pipeline_query,
            0,
            vk::QueryControlFlags::empty(),
        );
        device.cmd_draw_indexed_indirect(
            cmd,
            state.draw_indirect_buffer.buffer,
            0,
            renderables_count as u32,
            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
        );
        device.cmd_end_query(cmd, frame_data.pipeline_query, 0);

        device.cmd_end_rendering(cmd);
    };

    mew::giga_barrier(&device, cmd);

    // copy to swapchain
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

    // destroy outdated resources
    mew::destroy_image(device, &mut state.engine.allocator, &mut state.draw_image);
    mew::destroy_image(device, &mut state.engine.allocator, &mut state.depth_image);

    // update resources
    state.draw_image = mew::create_image(
        &device,
        &mut state.engine.allocator,
        state.engine.swapchain.extent,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::ImageAspectFlags::COLOR,
        false,
    );

    state.depth_image = mew::create_image(
        &device,
        &mut state.engine.allocator,
        state.engine.swapchain.extent,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        false,
    );

    // update descriptor info
    let mut image_infos: Vec<vk::DescriptorImageInfo> =
        Vec::from([vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(state.draw_image.view)]);
    state.engine.swapchain.views.iter().for_each(|image_view| {
        image_infos.push(
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(*image_view),
        );
    });

    let storage_image_descriptor_counts = image_infos.len();

    // update descriptors
    let storage_descriptor_write = vk::WriteDescriptorSet::default()
        .dst_set(state.descriptor_sets[1])
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&image_infos)
        .descriptor_count(storage_image_descriptor_counts as u32);
    unsafe {
        device.update_descriptor_sets(&[storage_descriptor_write], &[]);
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
