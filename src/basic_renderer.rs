use ash::vk;
use glam::Mat4;

// TODO: split rendering techniques into modules, where modules store their push constant data and resources they create
// dump each renderer modules into top level renderer that wraps everything into a giant struct?

pub struct CullRenderer {
    pub constants: CullConstants,
}

#[repr(C)]
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

#[repr(C)]
pub struct MeshConstants {
    pub view_proj: Mat4,
    pub vertex_buffer: vk::DeviceAddress,
    pub mesh_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
}

pub struct CopySwapchainRenderer {
    pub constants: CopySwapchainConstants,
}

#[repr(C)]
pub struct CopySwapchainConstants {
    pub src_id: u32,
    pub dst_id: u32,
}
