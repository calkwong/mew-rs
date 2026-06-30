#![allow(non_camel_case_types)]

type float3 = glam::Vec3;
type mat4x4 = glam::Mat4;

pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub nodes: Vec<Node>,
    pub node_transforms: Vec<NodeTransform>,
}

pub struct Node {
    mesh: MeshAsset,
    children: Vec<usize>,
}

pub struct NodeTransform {
    local_transform: mat4x4,
    world_transform: mat4x4,
}

#[derive(Copy, Clone)]
pub struct MeshAsset {
    mesh: u32,
    count: u32,
}

pub struct Mesh {
    vertex_offset: u32,
}

pub struct Vertex {
    pos: float3,
    normal: float3,
}

fn get_indices(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    out_buffer: &mut Vec<u32>,
    accessor_index: usize,
) {
    let accessor = &gltf.accessors[accessor_index];
    let buffer_view = &gltf.buffer_views[accessor.buffer_view.unwrap()];

    let accessor_byte_offset = &accessor.byte_offset;
    let buffer_view_byte_offset = buffer_view.byte_offset;

    let start = accessor_byte_offset + buffer_view_byte_offset;
    let count = accessor.count;

    let byte_size = accessor.component_type.byte_size();
    assert_eq!(byte_size, 2);
    let byte_stride = buffer_view.byte_stride.unwrap_or(byte_size);

    // For global combined index buffer
    let index_offset = out_buffer.len() as u32;

    out_buffer.extend((0..count).map(|i| {
        let vertex_offset = start + i * byte_stride;

        let index = u16::from_le_bytes(
            buffer_data[vertex_offset..vertex_offset + byte_size]
                .try_into()
                .unwrap(),
        ) as u32;

        index + index_offset
    }));
}

fn get_positions(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    out_buffer: &mut Vec<float3>,
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

        float3 { x, y, z }
    }))
}

// TODO: turn into get_float3 instead?
fn get_normals(
    gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
    buffer_data: &[u8],
    out_buffer: &mut Vec<float3>,
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

        float3 { x, y, z }
    }))
}

pub fn load_gltf(path: &str) -> Scene {
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
    // let mut meshes: Vec<Vec<Mesh>> = Vec::new();
    let mut mesh_assets: Vec<MeshAsset> = Vec::new();
    let mut meshes: Vec<Mesh> = Vec::new();

    let mut indices: Vec<u32> = Vec::new();
    let mut positions: Vec<float3> = Vec::new();
    let mut normals: Vec<float3> = Vec::new();

    // We need a threadsafe container for parallel loading?
    for m in &gltf.meshes {
        let mesh: MeshAsset = MeshAsset {
            mesh: meshes.len() as u32,
            count: m.primitives.len() as u32,
        };

        for primitive in &m.primitives {
            let indices_idx = primitive.indices.unwrap();
            get_indices(&gltf, buffer_data, &mut indices, indices_idx);
            // indices.iter().for_each(|index| println!("{index}"));

            // For global combined index buffer
            meshes.push(Mesh {
                vertex_offset: positions.len() as u32,
            });

            let positions_idx = primitive.attributes.position.unwrap();
            get_positions(&gltf, buffer_data, &mut positions, positions_idx);
            // positions.iter().for_each(|pos| println!("{}", pos));

            let normal_idx = primitive.attributes.normal.unwrap();
            get_normals(&gltf, buffer_data, &mut normals, normal_idx);
            // normals.iter().for_each(|normal| println!("{}", normal));

            assert_eq!(positions.len(), normals.len());
        }

        mesh_assets.push(mesh);
    }

    // May need to move into loop above
    let iter = std::iter::zip(positions, normals);
    let vertices: Vec<Vertex> = iter.map(|(pos, normal)| Vertex { pos, normal }).collect();

    // Build nodes
    let mut nodes: Vec<Node> = Vec::new();
    let mut node_transforms: Vec<NodeTransform> = Vec::new();
    for gltf_node in &gltf.nodes {
        if gltf_node.mesh.is_none() {
            continue;
        }

        if let Some(index) = gltf_node.mesh {
            let new_node = Node {
                mesh: mesh_assets[index],
                children: gltf_node.children.clone(),
            };

            nodes.push(new_node);
        }

        let mut local_transform = mat4x4::IDENTITY;

        match (
            gltf_node.matrix,
            gltf_node.translation,
            gltf_node.rotation,
            gltf_node.scale,
        ) {
            (Some(matrix), None, None, None) => {
                local_transform = mat4x4::from_cols_array(&matrix);
            }
            (_, Some(translation), Some(rotation), Some(scale)) => {
                let scale = glam::vec3(scale[0], scale[1], scale[2]);
                // TODO: verify not wxyz
                let rotation =
                    glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
                let translation = glam::vec3(translation[0], translation[1], translation[2]);

                local_transform =
                    mat4x4::from_scale_rotation_translation(scale, rotation, translation);
            }
            _ => {}
        };

        node_transforms.push(NodeTransform {
            local_transform,
            world_transform: mat4x4::IDENTITY,
        });
    }

    // Apply parent-child transforms
    for child_index in &gltf.scenes[0].nodes {
        refresh_transform(&nodes, &mut node_transforms, *child_index, mat4x4::IDENTITY);
    }

    Scene {
        vertices,
        indices,
        nodes,
        node_transforms,
    }
}

fn refresh_transform(
    nodes: &Vec<Node>,
    node_transforms: &mut Vec<NodeTransform>,
    index: usize,
    parent_matrix: mat4x4,
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
