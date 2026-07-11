// Passes must already be in topological order when registered with the rendergraph
// Supports automatic barriers, pass merging (partial reordering) and pass culling
// Does not support resource aliasing or arbitrary pass reordering
#![allow(dead_code)]
use ash::vk;
use std::{collections::HashMap, marker::PhantomData};

use crate as mew;
use mew::Image;

pub struct Rendergraph<'a> {
    resource_to_id: HashMap<&'a str, usize>,
    resource_states: Vec<ResourceState<'a>>,
    dependencies: Vec<Vec<usize>>,
    execution_groups: Vec<Vec<usize>>,
    executes: Vec<Box<dyn Fn(&ash::Device, vk::CommandBuffer, vk::PipelineLayout) + 'a>>,
    images: Vec<vk::Image>,
    depth_image: Option<vk::Image>,
}

struct ResourceState<'a> {
    name: &'a str,
    last_write: Option<usize>,
    reads_since_last_write: Vec<usize>,
}

impl<'a> Rendergraph<'a> {
    pub fn new() -> Self {
        Self {
            resource_to_id: HashMap::new(),
            resource_states: Vec::new(),
            dependencies: Vec::new(),
            execution_groups: Vec::new(),
            executes: Vec::new(),
            images: Vec::new(),
            depth_image: None,
        }
    }

    // The dependencies list built may contain duplicates, but dfs search ensures
    // we don't traverse the children of a node we have already visited via memoization
    pub fn add_pass<T>(&mut self, pass: Pass<'a, T>) {
        let pass_id = self.dependencies.len();
        self.dependencies.push(Vec::new());

        for PassResource { name, image } in &pass.reads {
            // Record dependencies
            if let Some(id) = self.resource_to_id.get(name) {
                self.resource_states[*id]
                    .reads_since_last_write
                    .push(pass_id);
                if let Some(last_write) = self.resource_states[*id].last_write {
                    self.dependencies[pass_id].push(last_write);
                }
            } else {
                self.resource_to_id.insert(name, self.resource_states.len());
                self.resource_states.push(ResourceState {
                    name: *name,
                    last_write: None,
                    reads_since_last_write: vec![pass_id],
                });
                if let Some(image_metadata) = image {
                    match (*image_metadata).aspect {
                        vk::ImageAspectFlags::DEPTH => {
                            self.depth_image = Some((*image_metadata).image);
                        }
                        vk::ImageAspectFlags::COLOR => {
                            self.images.push((*image_metadata).image);
                        }
                        _ => {
                            panic!("This should not execute!");
                        }
                    }
                }
            }
        }

        for PassResource { name, image } in &pass.writes {
            if let Some(id) = self.resource_to_id.get(name) {
                self.resource_states[*id].last_write = Some(pass_id);

                // Record dependencies, skip self to prevent deadlock
                for read in &self.resource_states[*id].reads_since_last_write {
                    if *read != pass_id {
                        self.dependencies[pass_id].push(*read);
                    }
                }
            } else {
                self.resource_to_id.insert(name, self.resource_states.len());
                self.resource_states.push(ResourceState {
                    name: *name,
                    last_write: Some(pass_id),
                    reads_since_last_write: Vec::new(),
                });
                if let Some(image_metadata) = image {
                    match (*image_metadata).aspect {
                        vk::ImageAspectFlags::DEPTH => {
                            self.depth_image = Some((*image_metadata).image);
                        }
                        vk::ImageAspectFlags::COLOR => {
                            self.images.push((*image_metadata).image);
                        }
                        _ => {
                            panic!("This should not execute!");
                        }
                    }
                }
            }
        }

        self.executes.push(pass.execute);
    }

    fn build_execution_groups(&mut self) {
        let len = self.dependencies.len();
        let root_index = len - 1;

        let mut dependency_levels: Vec<Vec<usize>> = vec![Vec::new(); len];
        let mut dependency_map: Vec<Option<i32>> = vec![None; len];

        self.dfs(root_index, &mut dependency_levels, &mut dependency_map);
        self.execution_groups = dependency_levels;
    }

    fn dfs(
        &self,
        index: usize,
        dependency_levels: &mut Vec<Vec<usize>>,
        dependency_map: &mut Vec<Option<i32>>,
    ) -> i32 {
        let mut level = -1;

        let dependencies = &self.dependencies[index];
        for dep in dependencies {
            // Memoization
            if let Some(dep_level) = dependency_map[*dep] {
                level = level.max(dep_level);
            } else {
                let dep_level = self.dfs(*dep, dependency_levels, dependency_map);
                level = level.max(dep_level);
            }
        }
        level = level + 1;
        dependency_map[index] = Some(level);
        dependency_levels[level as usize].push(index);
        level
    }

    pub fn compile(&mut self) {
        self.build_execution_groups();
    }

    pub fn run(&self, device: &ash::Device, cmd: vk::CommandBuffer, layout: vk::PipelineLayout) {
        let mut image_memory_barriers: Vec<vk::ImageMemoryBarrier2> = self
            .images
            .iter()
            .map(|image| {
                vk::ImageMemoryBarrier2::default()
                    .image(*image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .subresource_range(mew::image_subresource_range(vk::ImageAspectFlags::COLOR))
            })
            .collect();

        if let Some(depth) = self.depth_image {
            image_memory_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .image(depth)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .subresource_range(mew::image_subresource_range(vk::ImageAspectFlags::DEPTH)),
            )
        }

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

        for group in &self.execution_groups {
            if group.len() > 0 {
                for execute_id in group {
                    (self.executes[*execute_id])(device, cmd, layout);
                }
            } else {
                break;
            }
            mew::giga_barrier(device, cmd);
        }

        // dbg!(&self.dependencies);
        // dbg!(&self.execution_groups);
    }
}

pub enum RenderPass {
    Compute(ComputePass),
    Graphics(GraphicsPass),
}

#[derive(Copy, Clone)]
struct ImageMetadata {
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
}

pub struct PassResource<'a> {
    name: &'a str,
    image: Option<ImageMetadata>,
}

pub struct Pass<'a, T> {
    reads: Vec<PassResource<'a>>,
    writes: Vec<PassResource<'a>>,
    pipeline: vk::Pipeline,
    constants: &'a [u8],
    // TODO: Fn or FnOnce?
    execute: Box<dyn Fn(&ash::Device, vk::CommandBuffer, vk::PipelineLayout) + 'a>,
    render_pass: RenderPass,
    _marker: PhantomData<T>,
}

impl<'a, T> Pass<'a, T> {
    fn new(render_pass: RenderPass) -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            pipeline: vk::Pipeline::null(),
            constants: &[],
            execute: Box::new(|_, _, _| {}),
            render_pass,
            _marker: PhantomData,
        }
    }

    pub fn read_buffer(mut self, name: &'a str) -> Self {
        self.reads.push(PassResource { name, image: None });
        self
    }

    // TODO: handle temporal texture ie. TAA History
    pub fn read_image(
        mut self,
        name: &'a str,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
        self.reads.push(PassResource {
            name,
            image: Some(ImageMetadata { image, aspect }),
        });
        self
    }

    // This implicitly handles WAW
    // TODO: if we automate with fine grained barriers, reevaluate how our handling of WAW could affect access_mask; for a gigabarrier this is fine
    pub fn write_buffer(mut self, name: &'a str) -> Self {
        self.writes.push(PassResource { name, image: None });
        self.reads.push(PassResource { name, image: None });
        self
    }

    pub fn write_image(
        mut self,
        name: &'a str,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
        let image_metadata = ImageMetadata { image, aspect };

        self.writes.push(PassResource {
            name,
            image: Some(image_metadata),
        });
        self.reads.push(PassResource {
            name,
            image: Some(image_metadata),
        });

        self
    }

    pub fn constants(mut self, data: &'a [u8]) -> Self {
        self.constants = data;
        self
    }

    pub fn pipeline(mut self, pipeline: vk::Pipeline) -> Self {
        self.pipeline = pipeline;
        self
    }
}

// TODO: dispatch_indirect variant
pub struct ComputePass {
    x: u32,
    y: u32,
    z: u32,
}

impl<'a> Pass<'a, ComputePass> {
    pub fn new_compute() -> Self {
        Pass::new(RenderPass::Compute(ComputePass { x: 0, y: 0, z: 0 }))
    }

    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        match self.render_pass {
            RenderPass::Compute(_) => {}
            _ => panic!("Dispatch called on wrong render pass type"),
        }

        self.execute = Box::new(move |device, cmd, layout| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::ALL, 0, self.constants);
            device.cmd_dispatch(cmd, x, y, z);
        });

        self
    }
}

pub struct AttachmentDesc {
    pub view: vk::ImageView,
    pub load_op: vk::AttachmentLoadOp,
}

// TODO: draw, draw_mesh_task variant variants
pub struct GraphicsPass {
    render_targets: Vec<AttachmentDesc>,
    depth_target: Option<AttachmentDesc>,
}

impl<'a> Pass<'a, GraphicsPass> {
    pub fn new_graphics() -> Self {
        Pass::new(RenderPass::Graphics(GraphicsPass {
            render_targets: Vec::new(),
            depth_target: None,
        }))
    }

    pub fn render_target(mut self, name: &'a str, image: &Image, load_op: vk::AttachmentLoadOp) -> Self {
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            data.render_targets.push(AttachmentDesc {
                view: image.view,
                load_op,
            });
        }
        self = self.write_image(name, image.image, vk::ImageAspectFlags::COLOR);
        self
    }

    pub fn depth_target(mut self, name: &'a str, image: &Image, load_op: vk::AttachmentLoadOp) -> Self {
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            data.depth_target = Some(AttachmentDesc {
                view: image.view,
                load_op,
            });
        }
        self = self.write_image(name, image.image, vk::ImageAspectFlags::DEPTH);
        self
    }

    pub fn draw_indirect(
        mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
        index_buffer: vk::Buffer,
        extent: vk::Extent2D,
    ) -> Self {
        let mut color_attachments: Vec<vk::RenderingAttachmentInfo> = Vec::new();
        let mut depth_attachment: Option<vk::RenderingAttachmentInfo> = None;
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            color_attachments = data
                .render_targets
                .iter()
                .map(|target| {
                    let mut attachment = vk::RenderingAttachmentInfo::default()
                        .image_view(target.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(target.load_op)
                        .store_op(vk::AttachmentStoreOp::STORE);

                    if target.load_op == vk::AttachmentLoadOp::CLEAR {
                        attachment = attachment.clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        });
                    }

                    attachment
                })
                .collect();

            if let Some(depth_target) = &data.depth_target {
                let mut attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_target.view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(depth_target.load_op)
                    .store_op(vk::AttachmentStoreOp::STORE);

                if depth_target.load_op == vk::AttachmentLoadOp::CLEAR {
                    attachment = attachment.clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    });
                }

                depth_attachment = Some(attachment);
            }
        }

        let viewport = [vk::Viewport::default()
            .y(extent.height as f32)
            .width(extent.width as f32)
            .height(-(extent.height as f32))
            .max_depth(1.0)
            .min_depth(0.0)];

        let scissors = [vk::Rect2D {
            extent,
            ..Default::default()
        }];

        self.execute = Box::new(move |device, cmd, layout| unsafe {
            let mut rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    extent,
                    ..Default::default()
                })
                .color_attachments(&color_attachments)
                .layer_count(1);

            if let Some(ref depth_attachment) = depth_attachment {
                rendering_info = rendering_info.depth_attachment(depth_attachment);
            }

            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(cmd, 0, &viewport);
            device.cmd_set_scissor(cmd, 0, &scissors);
            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::ALL, 0, self.constants);
            device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed_indirect_count(
                cmd,
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            );
            device.cmd_end_rendering(cmd);
        });

        self
    }
}
