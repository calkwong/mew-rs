// Passes must already be in topological order when registered with the rendergraph
// Supports automatic barriers, pass merging (partial reordering) and pass culling
// Does not support resource aliasing or arbitrary pass reordering
#![allow(dead_code)]
use ash::vk;
use std::{collections::HashMap, marker::PhantomData};

use crate as mew;
use mew::Image;

pub struct Graph<'a> {
    resource_to_id: HashMap<&'a str, usize>,
    resource_states: Vec<ResourceState<'a>>,
    dependencies: Vec<Vec<usize>>,
    execution_groups: Vec<Vec<usize>>,
    executes: Vec<RenderPass>,
}

struct ResourceState<'a> {
    name: &'a str,
    last_write: Option<usize>,
    reads_since_last_write: Vec<usize>,
}

impl<'a> Graph<'a> {
    pub fn new() -> Self {
        Self {
            resource_to_id: HashMap::new(),
            resource_states: Vec::new(),
            dependencies: Vec::new(),
            execution_groups: Vec::new(),
            executes: Vec::new(),
        }
    }

    pub fn add_pass<T>(&mut self, pass: Pass<'a, T>) {
        let pass_id = self.dependencies.len();
        self.dependencies.push(Vec::new());

        for name in &pass.reads {
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
            }
        }

        for name in &pass.writes {
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
            }
        }

        self.executes.push(pass.render_pass);
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

    pub fn run(&self) {
        for group in &self.execution_groups {
            if group.len() > 0 {
                for execute_id in group {
                    match self.executes[*execute_id] {
                        // TODO:
                        RenderPass::Compute(_) => {}
                        RenderPass::Graphics(_) => {}
                    }
                }
            } else {
                break;
            }
        }
    }
}

pub enum RenderPass {
    Compute(ComputePass),
    Graphics(GraphicsPass),
}

pub struct Pass<'a, T> {
    reads: Vec<&'a str>,
    writes: Vec<&'a str>,
    // TODO: this is fragile, can we do better than inlining 256 bytes?
    constants: &'a [u8],
    render_pass: RenderPass,
    _marker: PhantomData<T>,
}

impl<'a, T> Pass<'a, T> {
    pub fn read(mut self, name: &'a str) -> Self {
        self.reads.push(name);
        self
    }

    pub fn write(mut self, name: &'a str) -> Self {
        self.writes.push(name);

        // This handles WAW
        // TODO: if we automate with fine grained barriers, reevaluate how this affects access_mask; for a gigabarrier this is fine
        self.reads.push(name);
        self
    }

    pub fn constants(mut self, data: &'a [u8]) -> Self {
        self.constants = data;
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
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            constants: &[],
            render_pass: RenderPass::Compute(ComputePass { x: 0, y: 0, z: 0 }),
            _marker: PhantomData,
        }
    }

    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        if let RenderPass::Compute(data) = &mut self.render_pass {
            data.x = x;
            data.y = y;
            data.z = z;
        } else {
            panic!("Dispatch called on wrong render pass type");
        }
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
    custom: Box<dyn Fn()>,
}

impl<'a> Pass<'a, GraphicsPass> {
    pub fn new_graphics() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            constants: &[],
            render_pass: RenderPass::Graphics(GraphicsPass {
                render_targets: Vec::new(),
                depth_target: None,
                custom: Box::new(|| {}),
            }),
            _marker: PhantomData,
        }
    }

    pub fn render_target(mut self, image: &Image, load_op: vk::AttachmentLoadOp) -> Self {
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            data.render_targets.push(AttachmentDesc {
                view: image.view,
                load_op,
            });
        } else {
            panic!("Render target called on wrong render pass type");
        }
        self
    }

    pub fn depth_target(mut self, image: &Image, load_op: vk::AttachmentLoadOp) -> Self {
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            data.depth_target = Some(AttachmentDesc {
                view: image.view,
                load_op,
            });
        } else {
            panic!("Depth target called on wrong render pass type");
        }
        self
    }

    pub fn custom<F>(mut self, f: F) -> Self
    where
        F: Fn() + 'static,
    {
        if let RenderPass::Graphics(data) = &mut self.render_pass {
            data.custom = Box::new(f);
        } else {
            panic!("Custom called on wrong render pass type");
        }
        self
    }
}
