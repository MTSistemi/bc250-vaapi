# Release v0.5.1: Gaming Mode Fixes, Steam Link Single-Slice, 16ms GPU Shield & Audio Parity

Release **v0.5.1** is a major stability and performance update for the **AMD BC-250 (Cyan Skillfish)** APU. It resolves game-streaming black screens in Steam Link and Sunshine/Moonlight across both Desktop and Gaming Modes (Gamescope), eliminates GPU queue latency spikes under 100% 3D game contention (e.g. *Red Dead Redemption 2*), resolves kernel audio driver clashes on CachyOS 7.2+, and introduces high-throughput parallel multi-slice transcoding.

### 🚀 Key Highlights & Critical Improvements

#### 1. Gaming Mode & Steam Link / Sunshine Streaming Stabilization
* **Single-Slice Stream Enforcement (Desktop & Gaming Mode)**: Hardware decoders on smart TVs, Android TV devices, and Steam Link client hardware do not support multi-slice H.264 streams or discard subsequent slices within a frame, rendering a solid grey or black screen. The driver now detects Steam Remote Play callers (`steam`, `streaming_client`, `steamwebhelper`) and Gamescope sessions (`gamescope`), automatically enforcing single-slice streams (`num_slices = 1` and `b_sliced_threads = 0`) across both compute and x264 backends.
* **External Buffer Fallback Handling**: Updated `bc250_CreateSurfaces2()` to process the `VASurfaceAttribExternalBufferDescriptor` attribute enum and explicitly reject unsupported external buffer memory descriptors with `VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE`. This prompts Gamescope and Sunshine to cleanly fall back to supported EGL / shared memory frame capture pathways.
* **Surface Attributes Advertisement**: Updated `bc250_QuerySurfaceAttributes()` to explicitly advertise `VASurfaceAttribMemoryType = VA_SURFACE_ATTRIB_MEM_TYPE_VA`.
* **Gaming Mode Screencasting & KMS Permissions**: Updated `tools/bc250_diagnose.sh` to detect active Gamescope sessions and verify Sunshine's `CAP_SYS_ADMIN` capability (required for DRM KMS capture under Linux `AT_SECURE`).
* **CABAC Chroma Intra Prediction Fix**: Fixed `encode_mb_i16x16_cabac()` to correctly transmit sanitized `chroma_pred_mode`, preventing chroma decoding corruption and visual glitches in CABAC streams.

#### 2. Heavy 3D Game Contention Shield & 16ms Bound (RDR 2 Benchmark)
* **16ms Bounded GPU Fence Wait**: Titles that saturate 100% of the 40 Compute Units (such as *Red Dead Redemption 2* running on DirectX 12 / Vulkan via Proton) submit massive rendering command buffers to the GPU. Previously, unbounded fence waits (`UINT64_MAX`) in `gpu_compute_sync_slot()` and `gpu_compute_begin_picture()` blocked the encoder thread for up to 200 ms while waiting for GPU compute passes to schedule behind graphics queues. Waits are now strictly bounded to 16 ms (1 frame interval at 60 fps) for live streaming callers (`sunshine`, `steam`, `streaming_client`, `wivrn`, `gamescope`) or configurable via `BC250_GPU_TIMEOUT_MS`.
* **Frame-Path Retry Truncation (1260ms to 6ms)**: Replaced the 7-attempt exponential retry sleep schedule in `gpu_compute_create_image()` and `gpu_compute_end_picture()` with a 3-attempt 6 ms ceiling for live streaming callers, preventing transient driver back-pressure from freezing live streams.
* **Race-Free CPU SIMD ME Staging**: Gated Tier 2 CPU SIMD motion estimation reads with `gpu_compute_wait_for_image_ready_host()` and `gpu_compute_cpu_read_safe()`, and added row-by-row streaming staging (`me_src_stage`/`me_ref_stage`) into cached host memory. Eliminates uncached GART aperture read bottlenecks and fixes the race condition that caused garbled video during fast action in game benchmarks.
* **Latency Flutter Shield & Dwell Gating**: Added `min_dwell_frames = 8` hysteresis to prevent erratic jumping into Tier 2 on isolated spike frames, and made Tier 3 failover step down to Tier 1 rather than Tier 2 to preserve smooth frame pacing.
* **Mesa gfx10 Legal CU Ring-Fencing**: Refined `bc250_pick_cu_split()` in `gpu_compute.c` and `tools/sunshine_preset/apply_gpu_ringfence.sh` to discover non-contiguous CU masks that strictly conform to Mesa's gfx10 late-alloc legality rules (requiring CU2/CU3 in each mask).

#### 3. High-Performance CPU x264 Backend & 32-bit Multilib Companion
* **Zero-GPU CPU x264 Streaming**: Integrated `libx264` backend (`BC250_H264_BACKEND=x264`) executes exclusively on Zen 2 CPU cores with 0% GPU overhead, achieving steady 2–4 ms encode latency during intense 100% GPU-bound 3D gaming.
* **Command-Line Preset Interception**: Intercepts CLI `-preset` and `-crf` parameters directly from `/proc/self/cmdline` alongside `BC250_PRESET` and `X264_PRESET` overrides. Switches live callers to `ultrafast` and caps offline transcodes to 4 threads (`threads 4`) to prevent CPU saturation.
* **32-bit Companion Driver**: Automated 32-bit driver build (`bc250-driver-linux-i386.tar.gz`) with multilib and 32-bit `libx264` support for Steam Link and 32-bit Wine applications.

#### 4. High-Throughput Multi-Slice Parallelization (Offline Transcoding)
* **OpenMP Parallel Slices**: For FFmpeg offline video transcoding, the compute encoder defaults to 4 parallel slices per frame processed concurrently across OpenMP worker threads.
* **4x Throughput Acceleration**: Slashes entropy coding latency from ~50 ms down to ~10–12 ms, pushing encode throughput beyond 100 FPS (>4x realtime).
* **Full CLI Integration**: Fully honors `-threads` and `-slices` from FFmpeg CLI, as well as `BC250_SLICES_PER_FRAME`.

#### 5. Audio Subsystem & CachyOS 7.2+ Kernel Parity (Issue #54)
* **Native HDMI/DP Audio Detection**: Updated `tools/bc250_diagnose.sh` to check for native HDMI/DP audio endpoints (`hdmi_devs > 0`) provided by modern kernels (CachyOS 7.2+).
* **Conflict Prevention**: Deprecates conflicting legacy DKMS `bc250_audio_fix` module when native endpoints are present, preventing audio subsystem clashes on modern kernels.

#### 6. Security Hardening & Architectural Precision (PR #56)
* **CodeQL World-Writable Creation Fix**: Explicitly enforced safe file modes (`0600` with `O_NOFOLLOW` in tests, `0644` in diagnostic tools) resolving CodeQL alerts (PR #56).
* **Semi-Custom RDNA 1.5 Architecture Documentation**: Documented BC-250 / Cyan Skillfish APU (`gfx1013`) as semi-custom Oberon RDNA 1.5 (RDNA 2 compute units and ray-tracing BVH, RDNA 1 memory architecture without Infinity Cache).

### 📦 Installation & Bundled Assets

**Pre-built Release Bundles:**
* 64-bit Driver: `bc250-driver-linux-x86_64.tar.gz` (installs to `/usr/local/lib64/dri/` or `/usr/lib/dri/`)
* 32-bit Companion Driver: `bc250-driver-linux-i386.tar.gz` (installs to `/usr/lib32/dri/` for Steam Link)

**Quickstart Installation:**
```bash
tar -xzf bc250-driver-linux-x86_64.tar.gz
sudo ./install.sh
```

**Arch Linux / CachyOS:**
```bash
./tools/install_cachyos_arch.sh --with-32bit
```

**SteamOS / HoloISO:**
```bash
sudo ./tools/setup_steamos.sh
```
