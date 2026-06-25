use ash::{Entry, Instance, ext::debug_utils, vk};
// use ash_window;

use std::ffi::c_char;

fn main() {
    let entry = Entry::linked();

    let app_name = c"mew-rust";

    let layer_names = [c"VK_LAYER_KHRONOS_validation"];
    let layers_names_raw: Vec<*const c_char> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut extension_names: Vec<*const c_char> = Vec::new();
    extension_names.push(debug_utils::NAME.as_ptr());

    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(vk::InstanceCreateFlags::default());

    unsafe {
        let instance: Instance = entry
            .create_instance(&instance_create_info, None)
            .expect("Instance creation failed");

        // destroys
        instance.destroy_instance(None);
    }
}
