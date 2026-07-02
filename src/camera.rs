use glam;
use glam::Vec3;

pub struct Camera {
    state: InputState,
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

pub enum Key {
    W,
    A,
    S,
    D,
}

pub enum KeyState {
    Pressed,
    Released,
}

#[derive(Default)]
struct InputState {
    key_w: bool,
    key_a: bool,
    key_s: bool,
    key_d: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            state: InputState::default(),
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

    pub fn process_input(&mut self, key: Key, state: KeyState) {
        match (key, state) {
            (Key::W, KeyState::Pressed) => self.state.key_w = true,
            (Key::A, KeyState::Pressed) => self.state.key_a = true,
            (Key::S, KeyState::Pressed) => self.state.key_s = true,
            (Key::D, KeyState::Pressed) => self.state.key_d = true,
            (Key::W, KeyState::Released) => self.state.key_w = false,
            (Key::A, KeyState::Released) => self.state.key_a = false,
            (Key::S, KeyState::Released) => self.state.key_s = false,
            (Key::D, KeyState::Released) => self.state.key_d = false,
        }

        self.velocity.x = (self.state.key_d as i32 - self.state.key_a as i32) as f32;
        self.velocity.z = (self.state.key_s as i32 - self.state.key_w as i32) as f32;
    }

    pub fn update(&mut self, delta_time: f32) {
        // TODO: make orientation part of Camera struct
        let orientation = self.get_orientation();
        self.position += orientation * (self.velocity * self.speed * delta_time);
    }
}
