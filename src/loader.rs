use crate as mew;
use ash::vk;
use glam::Vec4;
use glam::{Mat4, Vec2, Vec3};
use mew::Image;
use mew::create_sampled_image;

pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub meshes: Vec<Mesh>,
    pub renderables: Vec<ObjectData>,
    pub images: Vec<Image>,
    pub materials: Vec<MaterialData>,
    // For pure static scene, technically we don't need these
    pub nodes: Vec<Node>,
    pub node_transforms: Vec<NodeTransform>,
}

#[allow(dead_code)]
pub struct ObjectData {
    world_transform: Mat4,
    mesh_id: u32,
    material_id: u32,
}

#[allow(dead_code)]
pub struct MaterialData {
    base_color_factor: Vec4,
    diffuse_id: u32,
}

pub struct Node {
    mesh: Option<MeshAsset>,
    children: Vec<usize>, // child nodes
}

pub struct NodeTransform {
    local_transform: Mat4,
    world_transform: Mat4,
}

#[derive(Copy, Clone)]
pub struct MeshAsset {
    mesh_id: u32,
    count: u32,
}

pub struct Mesh {
    pub vertex_offset: u32,
    pub first_index: u32,
    pub index_count: u32,
    pub radius: f32,
    pub center: Vec3,
}

#[allow(dead_code)]
pub struct Vertex {
    pos: Vec3,
    uv_x: f32,
    normal: Vec3,
    uv_y: f32,
}

fn get_indices(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    out_buffer: &mut Vec<u32>,
    accessor_index: usize,
) -> u32 {
    let accessor = &gltf.accessors[accessor_index];
    let buffer_view = &gltf.buffer_views[accessor.buffer_view.unwrap()];

    let accessor_byte_offset = &accessor.byte_offset;
    let buffer_view_byte_offset = buffer_view.byte_offset;

    let start = accessor_byte_offset + buffer_view_byte_offset;
    let count = accessor.count;

    let byte_size = accessor.component_type.byte_size();
    assert_eq!(byte_size, 2);
    let byte_stride = buffer_view.byte_stride.unwrap_or(byte_size);

    out_buffer.extend((0..count).map(|i| {
        let vertex_offset = start + i * byte_stride;

        u16::from_le_bytes(
            buffer_data[vertex_offset..vertex_offset + byte_size]
                .try_into()
                .unwrap(),
        ) as u32
    }));

    count as u32
}

fn get_positions(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    accessor_index: usize,
) -> Vec<Vec3> {
    let accessor = &gltf.accessors[accessor_index];
    let buffer_view = &gltf.buffer_views[accessor.buffer_view.unwrap()];

    let accessor_byte_offset = &accessor.byte_offset;
    let buffer_view_byte_offset = buffer_view.byte_offset;

    let start = accessor_byte_offset + buffer_view_byte_offset;
    let count = accessor.count;

    let byte_size = accessor.component_type.byte_size();
    assert_eq!(byte_size, 4);
    let byte_stride = buffer_view.byte_stride.unwrap_or(byte_size * 3);

    (0..count)
        .map(|i| {
            let vertex_offset = start + i * byte_stride;

            let get = |k: usize| {
                f32::from_le_bytes(
                    buffer_data[vertex_offset + byte_size * k..vertex_offset + byte_size * (k + 1)]
                        .try_into()
                        .unwrap(),
                )
            };

            let x = get(0);
            let y = get(1);
            let z = get(2);

            Vec3 { x, y, z }
        })
        .collect()
}

// TODO: turn into get_Vec3 instead?
fn get_normals(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    out_buffer: &mut Vec<Vec3>,
    accessor_index: usize,
) {
    let accessor = &gltf.accessors[accessor_index];
    let buffer_view = &gltf.buffer_views[accessor.buffer_view.unwrap()];

    let accessor_byte_offset = &accessor.byte_offset;
    let buffer_view_byte_offset = buffer_view.byte_offset;

    let start = accessor_byte_offset + buffer_view_byte_offset;
    let count = accessor.count;

    let byte_size = accessor.component_type.byte_size();
    assert_eq!(byte_size, 4);
    let byte_stride = buffer_view.byte_stride.unwrap_or(byte_size * 3);

    out_buffer.extend((0..count).map(|i| {
        let vertex_offset = start + i * byte_stride;

        let get = |k: usize| {
            f32::from_le_bytes(
                buffer_data[vertex_offset + byte_size * k..vertex_offset + byte_size * (k + 1)]
                    .try_into()
                    .unwrap(),
            )
        };

        let x = get(0);
        let y = get(1);
        let z = get(2);

        Vec3 { x, y, z }
    }))
}

fn get_uv(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    accessor_index: usize,
) -> Vec<Vec2> {
    let accessor = &gltf.accessors[accessor_index];
    let buffer_view = &gltf.buffer_views[accessor.buffer_view.unwrap()];

    let accessor_byte_offset = &accessor.byte_offset;
    let buffer_view_byte_offset = buffer_view.byte_offset;

    let start = accessor_byte_offset + buffer_view_byte_offset;
    let count = accessor.count;

    let byte_size = accessor.component_type.byte_size();
    assert_eq!(byte_size, 4);
    let byte_stride = buffer_view.byte_stride.unwrap_or(byte_size * 2);

    (0..count)
        .map(|i| {
            let vertex_offset = start + i * byte_stride;

            let get = |k: usize| {
                f32::from_le_bytes(
                    buffer_data[vertex_offset + byte_size * k..vertex_offset + byte_size * (k + 1)]
                        .try_into()
                        .unwrap(),
                )
            };

            let x = get(0);
            let y = get(1);

            Vec2 { x, y }
        })
        .collect()
}

pub fn load_gltf(
    path: &str,
    device: &mew::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
) -> Scene {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();

    assert!(buffer.is_none()); // What check is this?
    assert_eq!(gltf.buffers.len(), 1);

    let path = std::path::Path::new(&path);

    let mut buffer_file =
        std::fs::File::open(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    std::io::Read::read_to_end(&mut buffer_file, &mut buffer).unwrap();
    let buffer_data: &[u8] = &buffer;

    // Each mesh which holds a vector of primitives is mapped to a Node
    let mut mesh_assets: Vec<MeshAsset> = Vec::new();
    let mut meshes: Vec<Mesh> = Vec::new();

    let mut indices: Vec<u32> = Vec::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();

    // Read file -> decompress data -> transcode -> upload to GPU
    let images: Vec<Image> = gltf
        .images
        .iter()
        .map(|img| {
            let mut file = std::fs::File::open(
                path.with_file_name(img.uri.as_ref().unwrap())
                    .to_str()
                    .unwrap(),
            )
            .unwrap();
            let mut buf: Vec<u8> = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut buf).unwrap();
            let data: &[u8] = &buf;

            let ktx2 = ktx2::Reader::new(data).expect("Can't create reader");
            let header = ktx2.header();

            assert_eq!(
                header.supercompression_scheme.unwrap(),
                ktx2::SupercompressionScheme::Zstandard
            );
            assert_eq!(header.format, None);

            let mut data: Vec<u8> = Vec::with_capacity(
                ktx2.levels()
                    .map(|level| level.uncompressed_byte_length)
                    .sum::<u64>() as _,
            );

            let mut offsets = Vec::with_capacity(ktx2.levels().len());

            for (i, level) in ktx2.levels().enumerate() {
                offsets.push(data.len());

                let decompressed_data =
                    &zstd::bulk::decompress(level.data, level.uncompressed_byte_length as _)
                        .unwrap();

                let transcoder = basis_universal::LowLevelUastcTranscoder::new();
                let width = (header.pixel_width >> i).max(1);
                let height = (header.pixel_height >> i).max(1);

                let output = transcoder
                    .transcode_slice(
                        decompressed_data,
                        basis_universal::SliceParametersUastc {
                            num_blocks_x: width.div_ceil(4),
                            num_blocks_y: height.div_ceil(4),
                            has_alpha: true,
                            original_width: width,
                            original_height: height,
                        },
                        basis_universal::DecodeFlags::HIGH_QUALITY,
                        basis_universal::TranscoderBlockFormat::BC7,
                    )
                    .unwrap();

                data.extend_from_slice(&output);
            }

            let format = match ktx2.transfer_function().unwrap() {
                ktx2::TransferFunction::Linear => vk::Format::BC7_UNORM_BLOCK,
                ktx2::TransferFunction::SRGB => vk::Format::BC7_SRGB_BLOCK,
                _ => panic!("Not currently supported"),
            };

            create_sampled_image(
                &device.device,
                device.graphics_queue,
                device.frame_resources[0].command_pool,
                device.frame_resources[0].command_buffer,
                allocator,
                vk::Extent2D {
                    width: header.pixel_width,
                    height: header.pixel_height,
                },
                format,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
                &offsets,
                &data,
                Some(ktx2.levels().len()),
            )
        })
        .collect();

    let descriptor_handles: Vec<u32> = images
        .iter()
        .map(|img| device.register_image(img.view, mew::RenderResourceTag::Sampled))
        .collect();

    let mut materials: Vec<MaterialData> = Vec::new();
    for m in &gltf.materials {
        let diffuse_id = if let Some(tex) = &m.pbr_metallic_roughness.base_color_texture {
            let texture_index = tex.index;
            if let Some(basisu_tex) = gltf.textures[texture_index].extensions.khr_texture_basisu {
                descriptor_handles[basisu_tex.source]
            } else {
                todo!("Implement non-basisu textures");
            }
        } else {
            0
        };

        let mat = MaterialData {
            base_color_factor: Vec4::from_array(m.pbr_metallic_roughness.base_color_factor),
            diffuse_id,
        };
        materials.push(mat);
    }

    let mut material_ids: Vec<u32> = Vec::new();
    for m in &gltf.meshes {
        let mesh: MeshAsset = MeshAsset {
            mesh_id: meshes.len() as u32,
            count: m.primitives.len() as u32,
        };

        for primitive in &m.primitives {
            let indices_idx = primitive.indices.unwrap();
            let vertex_offset = positions.len() as u32;
            let first_index = indices.len() as u32;
            let index_count = get_indices(&gltf, buffer_data, &mut indices, indices_idx);

            let positions_idx = primitive.attributes.position.unwrap();
            let new_positions = get_positions(&gltf, buffer_data, positions_idx);

            let mut center = Vec3::default();
            new_positions.iter().for_each(|p| {
                center += p;
            });
            center /= new_positions.len() as f32;
            let mut radius: f32 = 0.0;
            new_positions.iter().for_each(|p| {
                radius = p.distance(center).max(radius);
            });

            positions.extend(new_positions);

            // TODO: don't modify in fn
            let normal_idx = primitive.attributes.normal.unwrap();
            get_normals(&gltf, buffer_data, &mut normals, normal_idx);

            if let Some(uv_idx) = primitive.attributes.texcoord_0 {
                uvs.extend(get_uv(&gltf, buffer_data, uv_idx));
            } else {
                uvs.resize(positions.len(), Vec2::default());
            }

            if let Some(mat) = primitive.material {
                material_ids.push(mat as _);
            } else {
                panic!("No material unsupported");
            }

            // For global combined index buffer
            meshes.push(Mesh {
                vertex_offset,
                first_index,
                index_count,
                radius,
                center,
            });
        }

        mesh_assets.push(mesh);
    }

    assert_eq!(positions.len(), normals.len());
    assert_eq!(positions.len(), uvs.len());
    let vertices: Vec<Vertex> = (0..positions.len())
        .map(|i| {
            let pos = positions[i];
            let normal = normals[i];
            let uv = uvs[i];

            Vertex {
                pos,
                uv_x: uv[0],
                normal,
                uv_y: uv[1],
            }
        })
        .collect();

    // Build all nodes
    let mut nodes: Vec<Node> = Vec::new();
    let mut node_transforms: Vec<NodeTransform> = Vec::new();
    for gltf_node in &gltf.nodes {
        let new_node = match gltf_node.mesh {
            Some(index) => Node {
                mesh: Some(mesh_assets[index]),
                children: gltf_node.children.clone(),
            },
            None => Node {
                mesh: None,
                children: gltf_node.children.clone(),
            },
        };
        nodes.push(new_node);

        let local_transform = if let Some(matrix) = gltf_node.matrix {
            Mat4::from_cols_array(&matrix)
        } else {
            let scale = gltf_node.scale.unwrap_or([1.0, 1.0, 1.0]);
            let rotation = gltf_node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let translation = gltf_node.translation.unwrap_or([0.0, 0.0, 0.0]);

            let scale = glam::vec3(scale[0], scale[1], scale[2]);
            let rotation =
                glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
            let translation = glam::vec3(translation[0], translation[1], translation[2]);

            Mat4::from_scale_rotation_translation(scale, rotation, translation)
        };

        node_transforms.push(NodeTransform {
            local_transform,
            world_transform: Mat4::IDENTITY,
        });
    }

    // Apply parent-child transforms
    for child_index in &gltf.scenes[0].nodes {
        refresh_transform(&nodes, &mut node_transforms, *child_index, Mat4::IDENTITY);
    }

    let mut renderables: Vec<ObjectData> = Vec::new();

    nodes.iter().enumerate().for_each(|(index, node)| {
        if let Some(mesh) = node.mesh {
            (0..mesh.count).for_each(|i| {
                renderables.push(ObjectData {
                    world_transform: node_transforms[index].world_transform,
                    mesh_id: mesh.mesh_id + i,
                    material_id: material_ids[(mesh.mesh_id + i) as usize],
                })
            });
        }
    });

    Scene {
        vertices,
        indices,
        meshes,
        renderables,
        images,
        materials,
        nodes,
        node_transforms,
    }
}

fn refresh_transform(
    nodes: &Vec<Node>,
    node_transforms: &mut Vec<NodeTransform>,
    index: usize,
    parent_matrix: Mat4,
) {
    node_transforms[index].world_transform = parent_matrix * node_transforms[index].local_transform;

    for child_index in &nodes[index].children {
        refresh_transform(
            nodes,
            node_transforms,
            *child_index,
            node_transforms[index].world_transform,
        );
    }
}
