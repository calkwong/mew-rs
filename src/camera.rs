use glam;
use glam::Vec3;

pub struct Camera {
    pub position: Vec3,
    velocity: Vec3,
    pitch: f32,
    yaw: f32,
    pub near: f32,
    pub far: f32,
    pub fovy: f32, // radians
    speed: f32,
    sensitivity: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::new(0.0, 0.0, 0.0),
            velocity: Vec3::new(0.0, 0.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
            near: 0.1,
            far: 100.0,
            fovy: 70.0_f32.to_radians(),
            speed: 5.0,
            sensitivity: 0.005,
        }
    }
}

impl Camera {
    pub fn position(mut self, position: glam::Vec3) -> Self {
        self.position = position;
        self
    }

    fn get_orientation(&self) -> glam::Quat {
        let pitch_rotation =
            glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 0.0, 0.0), self.pitch);
        let yaw_rotation = glam::Quat::from_axis_angle(glam::Vec3::new(0.0, -1.0, 0.0), self.yaw);

        pitch_rotation * yaw_rotation
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let inv_orientation = self.get_orientation().conjugate();
        let mut view = glam::Mat4::from_quat(inv_orientation);
        let rotated_pos = inv_orientation.mul_vec3(-self.position);
        view.w_axis = glam::Vec4::new(rotated_pos.x, rotated_pos.y, rotated_pos.z, 1.0);

        view
    }

    // TODO: broken
    pub fn process_input(&mut self, horizontal: f32, forward: f32) {
        self.velocity.x = horizontal;
        self.velocity.z = forward;
    }

    // TODO: deltatime
    pub fn update(&mut self) {
        // TODO: make orientation part of Camera struct
        let orientation = self.get_orientation();
        self.position += orientation * (self.velocity * self.speed);
    }
}
