use ash::vk;
use glam::{Mat4, Vec2};

#[derive(Default)]
pub struct OcclusionRenderer {
    pub cull_constants: CullConstants,
    pub render_constants: RenderConstants,
    pub depth_pyramid_constants: DepthPyramidConstants,
    pub tonemap_constants: TonemapConstants,
}

impl OcclusionRenderer {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
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

#[repr(C)]
#[derive(Default)]
pub struct RenderConstants {
    pub view_proj: Mat4,
    pub vertex_buffer: vk::DeviceAddress,
    pub mesh_buffer: vk::DeviceAddress,
    pub object_buffer: vk::DeviceAddress,
    pub material_buffer: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Default)]
pub struct DepthPyramidConstants {
    pub spd_buffer: vk::DeviceAddress,
    pub rcp_resolution: Vec2,
    pub mips: u32,
    pub num_wgs: u32,
    pub src_id: u32,
    pub dst_id: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct TonemapConstants {
    pub src_id: u32,
    pub dst_id: u32,
}

// TODO:
// - set up renderpasses here and call it from main?
// - cleaner way of filling push constants?
