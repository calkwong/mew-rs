use ash::vk;
use glam::Mat4;

// TODO: split rendering techniques into modules, where modules store their push constant data and resources they create
// dump each renderer modules into top level renderer that wraps everything into a giant struct?

pub struct CullRenderer {
    pub constants: CullConstants,
}

impl CullRenderer {
    pub fn new() -> Self {
        Self {
            constants: CullConstants::default(),
        }
    }
}

impl Default for CullRenderer {
    fn default() -> Self {
        CullRenderer::new()
    }
}

#[repr(C)]
#[derive(Default)]
pub struct CullConstants {
    pub view: Mat4,
    pub mesh_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
    pub draw_indirect_buffer: vk::DeviceAddress,
    pub dispatch_buffer: vk::DeviceAddress,
    pub planes: glam::Vec4,
    pub p00: f32,
    pub p11: f32,
    pub near: f32,
    pub far: f32,
    pub count: u32,
}

pub struct MeshRenderer {
    pub constants: MeshConstants,
}

impl MeshRenderer {
    pub fn new() -> Self {
        Self {
            constants: MeshConstants::default(),
        }
    }
}

impl Default for MeshRenderer {
    fn default() -> Self {
        MeshRenderer::new()
    }
}

#[repr(C)]
#[derive(Default)]
pub struct MeshConstants {
    pub view_proj: Mat4,
    pub vertex_buffer: vk::DeviceAddress,
    pub mesh_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
}

pub struct CopyRenderer {
    pub constants: CopyConstants,
}

#[repr(C)]
#[derive(Default)]
pub struct CopyConstants {
    pub src_id: u32,
    pub dst_id: u32,
}

impl CopyRenderer {
    pub fn new() -> Self {
        Self {
            constants: CopyConstants::default(),
        }
    }
}

impl Default for CopyRenderer {
    fn default() -> Self {
        CopyRenderer::new()
    }
}
