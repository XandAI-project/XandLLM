use candle_core::Device;
use tracing::info;
use tracing::warn;

use crate::error::CoreResult;

/// Selects the best available compute device.
///
/// When the `cuda` feature is enabled and `prefer_gpu` is true, attempts to
/// acquire CUDA device `cuda_device_id`. Falls back to CPU on any error.
pub fn select_device(prefer_gpu: bool, #[allow(unused_variables)] cuda_device_id: usize) -> CoreResult<Device> {
    #[cfg(feature = "cuda")]
    if prefer_gpu {
        match Device::new_cuda(cuda_device_id) {
            Ok(dev) => {
                info!(cuda_device_id, "Using CUDA device");
                return Ok(dev);
            }
            Err(e) => {
                warn!(error = %e, "CUDA unavailable, falling back to CPU");
            }
        }
    }

    #[cfg(feature = "metal")]
    if prefer_gpu {
        match Device::new_metal(0) {
            Ok(dev) => {
                info!("Using Metal device");
                return Ok(dev);
            }
            Err(e) => {
                warn!(error = %e, "Metal unavailable, falling back to CPU");
            }
        }
    }

    info!("Using CPU device");
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    if prefer_gpu {
        warn!(
            "--gpu flag has no effect: binary was not compiled with GPU support. \
             Rebuild with `cargo install --path crates/xandllm-cli --features cuda` \
             (requires NVIDIA CUDA toolkit)."
        );
    }
    Ok(Device::Cpu)
}

/// Returns a human-readable description of a device.
pub fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_device_no_gpu_returns_cpu() {
        let device = select_device(false, 0).unwrap();
        assert!(
            matches!(device, Device::Cpu),
            "prefer_gpu=false must always return CPU"
        );
    }

    #[test]
    fn test_select_device_gpu_false_when_no_cuda_feature() {
        // Without the `cuda` or `metal` features compiled in, even prefer_gpu=true
        // should fall back to CPU cleanly.
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let device = select_device(true, 0).unwrap();
            assert!(matches!(device, Device::Cpu));
        }
    }

    #[test]
    fn test_device_name_cpu() {
        assert_eq!(device_name(&Device::Cpu), "CPU");
    }
}
