use gve_runtime::{Renderer, RenderConfig, create_renderer};
use wasm_bindgen::prelude::*;

pub struct WasmEngine {
    pub renderer: Renderer,
    pub surface: wgpu::Surface<'static>,
    pub adapter: wgpu::Adapter,
    #[allow(dead_code)]  // Stored for future adapter queries
    pub instance: wgpu::Instance,
    // Future: physics, audio, message_queue
}

// SAFETY: On WASM, we are limited to a single thread for now.
unsafe impl Send for WasmEngine {}
unsafe impl Sync for WasmEngine {}

impl WasmEngine {
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<Self, JsValue> {
        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Force WebGL2 (GL) to avoid WebGPU-specific limit negotiation errors
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        #[cfg(target_arch = "wasm32")]
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

        #[cfg(not(target_arch = "wasm32"))]
        let surface = instance.create_surface(canvas)
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;
        
        // SAFETY: On WASM, we know the instance and surface outlive the engine.
        // The surface needs to be 'static for the Renderer struct.
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        web_sys::console::log_1(&"Requesting WebGL2 adapter...".into());
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.ok_or_else(|| {
            let msg = "Failed to find any compatible graphics adapter (WebGL2).";
            web_sys::console::error_1(&msg.into());
            JsValue::from_str(msg)
        })?;

        let info = adapter.get_info();
        web_sys::console::log_1(&format!("Using adapter: {:?} ({:?})", info.name, info.backend).into());

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // Use minimum WebGL2 limits to ensure compatibility
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: Default::default(),
            },
            None,
        ).await.map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;



        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let renderer = create_renderer(device, queue, RenderConfig { width, height, surface_format });

        Ok(Self {
            renderer,
            surface,
            adapter,
            instance,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let mut config = self.surface.get_default_config(&self.adapter, width, height)
                .expect("Failed to get surface default config");
            
            // Re-apply optimizations or specific formats if needed
            config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
            
            self.surface.configure(&self.renderer.device, &config);
            self.renderer.resize(width, height);
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render_to_view(&view);

        output.present();
        Ok(())
    }
}
