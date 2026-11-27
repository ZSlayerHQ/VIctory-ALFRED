//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};

#[cfg(test)]
mod d16_root_tests;
#[cfg(test)]
mod link_tests;
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

/// Integer render target formats: D3D12 cannot blend into them.
fn integer_format(f: DXGI_FORMAT) -> bool {
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// Whether the host blends into render targets of `format`. It never does into integer
/// formats (`R8_UINT`...), where the console's hardware leaves blending out: D3D12 says
/// so for every host ([`integer_format`]), and the host is asked about the others.
fn blendable(format: DXGI_FORMAT) -> bool {
    static KNOWN: std::sync::Mutex<Vec<(i32, bool)>> = std::sync::Mutex::new(Vec::new());
    if integer_format(format) {
        return false;
    }
    if format == DXGI_FORMAT_UNKNOWN {
        return true;
    }
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, b)) = known.iter().find(|k| k.0 == format.0) {
        return b;
    }
    let Some(host) = host() else { return true };
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
    // SAFETY: a plain query into our own structure.
    let asked = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast(),
            size_of_val(&support) as u32,
        )
    };
    let b = asked.is_err() || support.Support1.0 & D3D12_FORMAT_SUPPORT1_BLENDABLE.0 != 0;
    if !b {
        tracing::info!(format = format.0, "the host does not blend into this format: blending off for such targets");
    }
    known.push((format.0, b));
    b
}

/// The title's blend state as the host takes it: blending off on the render targets
/// whose format the host does not blend into ([`blendable`]). The console ignores
/// blending there, and titles leave it on (the world sets it on an `R8_UINT` target,
/// the windshield wiper mask blends MAX into one), but D3D12 does not allow it:
/// vkd3d-proton and the AMD driver refuse such a pipeline with `E_INVALIDARG`, and its
/// draws are lost. With one description for every target, each target gets its own
/// copy so the others keep theirs.
fn host_blend(blend: &D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) -> D3D12_BLEND_DESC {
    blend_into(blend, targets, blendable)
}

/// [`host_blend`] of a graphics state, with `blendable` telling which formats the host
/// blends into.
#[cfg(test)]
fn blend_where(g: &GraphicsState, blendable: impl Fn(DXGI_FORMAT) -> bool) -> D3D12_BLEND_DESC {
    blend_into(&g.blend, &g.targets, blendable)
}

/// [`host_blend`] with `blendable` telling which formats the host blends into.
fn blend_into(
    blend: &D3D12_BLEND_DESC,
    targets: &D3D12_RT_FORMAT_ARRAY,
    blendable: impl Fn(DXGI_FORMAT) -> bool,
) -> D3D12_BLEND_DESC {
    let mut b = *blend;
    let n = (targets.NumRenderTargets as usize).min(8);
    let rt = |b: &D3D12_BLEND_DESC, i: usize| {
        if b.IndependentBlendEnable.as_bool() { b.RenderTarget[i] } else { b.RenderTarget[0] }
    };
    let off: Vec<usize> = (0..n)
        .filter(|&i| rt(&b, i).BlendEnable.as_bool() && !blendable(targets.RTFormats[i]))
        .collect();
    // (A logic operation keeps the shared description: it needs one.)
    if off.is_empty() || (0..n).any(|i| rt(&b, i).LogicOpEnable.as_bool()) {
        return b;
    }
    if !b.IndependentBlendEnable.as_bool() {
        b.IndependentBlendEnable = true.into();
        for i in 1..8 {
            b.RenderTarget[i] = b.RenderTarget[0];
        }
    }
    for i in off {
        b.RenderTarget[i].BlendEnable = false.into();
    }
    b
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::target_format(crate::resource::pc_format(
                            f.0 as u32,
                        )) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate
                )
            })
            .collect();
        let bounds = if self.depth_bounds {
            " depth-bounds"
        } else {
            ""
        };
        format!(
            "{:?} {} {:?} {:?}{bounds} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    fn depth1(&self) -> D3D12_DEPTH_STENCIL_DESC1 {
        let d = &self.depth;
        D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: d.DepthEnable,
            DepthWriteMask: d.DepthWriteMask,
            DepthFunc: d.DepthFunc,
            StencilEnable: d.StencilEnable,
            StencilReadMask: d.StencilReadMask,
            StencilWriteMask: d.StencilWriteMask,
            FrontFace: d.FrontFace,
            BackFace: d.BackFace,
            DepthBoundsTestEnable: self.depth_bounds.into(),
        }
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether the title's root SRVs are bound as one-view tables of the
/// buffer's host copy, whose reads past its end return zeros, rather than
/// as root descriptors, which have no size: a mesh shader's vertex fetch
/// has no record limit, and a read past the copy reaches whatever the host
/// placed after it (memory it has freed: the device is removed, upstream
/// PR #493). `VICTORY_ROOT_SRV_DIRECT=1` goes back to root descriptors.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root UAVs: one-view tables like the SRVs (or root descriptors with
    /// `VICTORY_ROOT_SRV_DIRECT=1`).
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// The slot table (`SLOT_STRIDES`, among `root_srvs`) is a one-view
    /// table, not a root descriptor: with `VICTORY_D16_INTEGER_VIEWS=1`,
    /// where the descriptor did not fit the root signature.
    pub(crate) strides_table: bool,
}

/// Pipelines refused because the slot table fitted their root signature
/// neither as a root descriptor nor as a one-view table
/// (`VICTORY_D16_INTEGER_VIEWS=1`), since the start.
static D16_TABLES_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See [`D16_TABLES_REFUSED`].
#[cfg(test)]
pub(crate) fn d16_tables_refused() -> u64 {
    D16_TABLES_REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// D3D12's limit on a root signature, in DWORDs.
const ROOT_DWORDS: u32 = 64;

/// With `VICTORY_D16_INTEGER_VIEWS=1`: brings `params` inside
/// [`ROOT_DWORDS`] where the slot table's root descriptors (2 DWORDs each)
/// take it past, by making them one-view tables (1 DWORD) stage by stage
/// until it fits, and marks those stages ([`Stage::strides_table`]). With the
/// switch a table-bound d16 typed load asks for the slot table, so a stage
/// whose signature had 63 DWORDs (Sol's RED review of the accuracy ports,
/// Q4) keeps its pipeline at 64. False if even that does not fit while the
/// slot table is the cause (the pipeline is refused, by name, and counted);
/// a signature past the limit without the slot table is left to fail as on
/// `world`.
// (Boxed: the parameters point at the ranges, which must stay where they
// are as the vector grows.)
#[allow(clippy::vec_box)]
fn fit_strides(
    params: &mut [D3D12_ROOT_PARAMETER1],
    ranges: &mut Vec<Box<D3D12_DESCRIPTOR_RANGE1>>,
    bound: &mut [Stage],
) -> bool {
    let cost = |p: &D3D12_ROOT_PARAMETER1| match p.ParameterType {
        // SAFETY: the union member the type names.
        D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => unsafe { p.Anonymous.Constants.Num32BitValues },
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE => 1,
        _ => 2,
    };
    let mut total: u32 = params.iter().map(cost).sum();
    if total <= ROOT_DWORDS {
        return true;
    }
    let mut ours = false;
    for stage in bound.iter_mut() {
        let Some(&(_, at)) = stage.root_srvs.iter().find(|r| r.0 == victory_shader::emit::SLOT_STRIDES) else {
            continue;
        };
        ours = true;
        if total <= ROOT_DWORDS {
            break;
        }
        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: u32::from(victory_shader::emit::SLOT_STRIDES),
            RegisterSpace: ROOT_CBV_SPACE,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        }));
        let Some(range) = ranges.last() else { return false };
        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**range;
        let p = &mut params[at as usize];
        *p = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: range },
            },
            ShaderVisibility: p.ShaderVisibility,
        };
        stage.strides_table = true;
        total -= 1;
    }
    total <= ROOT_DWORDS || !ours
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Has the geometry stage that completes the console's rectangles
    /// (`victory_shader::emit::rect_list_gs`): its rectangle draws go out
    /// as the title gives them, three vertices a rectangle.
    pub(crate) rects: bool,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Its depth, stencil, cull and blend state in short, for the frame
    /// listing (empty for compute).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) state: String,
    /// Its program's number for the lines by program, once asked
    /// (`VICTORY_GPU_STAMPS=2`, [`HostPipeline::program`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) program: std::sync::OnceLock<crate::stamps::Program>,
}

#[cfg(feature = "diagnostics")]
impl HostPipeline {
    /// The number the lines by program count its calls under: its shaders
    /// as the log prints them (`name`) and its stage (`VICTORY_GPU_STAMPS=2`).
    pub(crate) fn program(&self) -> crate::stamps::Program {
        *self.program.get_or_init(|| {
            use crate::stamps::ProgramStage;
            let stage = if self.compute {
                ProgramStage::Compute
            } else if self.mesh {
                ProgramStage::Mesh
            } else if self.stages.iter().any(|s| s.shader.stage == "pixel") {
                ProgramStage::Pixel
            } else {
                ProgramStage::Vertex
            };
            crate::stamps::program(stage, &self.name)
        })
    }
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        // For the report when the host device is removed: what this thread
        // has in the library from here on.
        super::library::label(format!("{shaders:?}"));
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)", shaders);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // The ranges table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                strides_table: false,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    let uav = kind == D3D12_ROOT_PARAMETER_TYPE_UAV;
                    // Ours, not the title's, stay root descriptors: the
                    // table of strides, the directory of title memory by
                    // address and its fault notes, the slot map's directory
                    // and pool.
                    let ours = if uav {
                        slot == victory_shader::emit::SLOT_FAULTS
                    } else {
                        slot == victory_shader::emit::SLOT_STRIDES
                            || slot == victory_shader::emit::SLOT_MEMORY
                            || slot == victory_shader::emit::slot_map::MAP_DIR
                            || slot == victory_shader::emit::slot_map::MAP_POOL
                    };
                    if (uav || kind == D3D12_ROOT_PARAMETER_TYPE_SRV) && !ours && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the
                        // reads (and a root UAV's writes), which a root
                        // descriptor leaves unbounded.
                        // (Our own table of strides stays a descriptor.)
                        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                            RangeType: if uav {
                                D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                            } else {
                                D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                            },
                            NumDescriptors: 1,
                            BaseShaderRegister: u32::from(slot),
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                            OffsetInDescriptorsFromTableStart: 0,
                        }));
                        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last()?;
                        out.push((slot, params.len() as u32));
                        params.push(D3D12_ROOT_PARAMETER1 {
                            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                                    NumDescriptorRanges: 1,
                                    pDescriptorRanges: range,
                                },
                            },
                            ShaderVisibility: visibility,
                        });
                        continue;
                    }
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(static_sampler(host, s, n as u32, visibility));
            }
            bound.push(stage);
        }
        if crate::descriptor::host::view_kind::on()
            && !fit_strides(&mut params, &mut ranges, &mut bound)
        {
            let n = D16_TABLES_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(?shaders, refused = n, "VICTORY_D16_INTEGER_VIEWS=1: pipelines refused for the slot table's root DWORDs");
            }
            return refuse(
                "VICTORY_D16_INTEGER_VIEWS=1: the slot table fits the root signature neither as a root view nor as a one-view table",
                shaders,
            );
        }
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, compute.is_some()) {
            Ok(root) => root,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host root signature refused");
                return None;
            }
        };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
                #[cfg(feature = "diagnostics")]
                program: std::sync::OnceLock::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let depth_bounds = if g.depth_bounds {
            if host.depth_bounds() {
                true
            } else {
                static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    tracing::warn!(
                        "a pipeline enables the depth bounds test, which the host GPU does not support: its draws pass at every depth"
                    );
                }
                false
            }
        } else {
            false
        };
        let state = g.describe();
        if let Some(ms) = mesh {
            // A mesh program whose outputs pass D3D12's 32 KiB is linked with this pixel
            // program ([`linked_as`]): the pipeline gets the two made for each other, which bind
            // as the shared translations do, and when the host refuses that pipeline, the
            // shared translations ([`mesh_pair_pipeline`]).
            let bytecode = |dxil: &[u8]| D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            };
            let ps = stages.get(1).map(|&(ps, _)| ps);
            // What the stream below is made of besides the two programs: the state, and
            // whether the host's depth bounds test is in it.
            let made_of = format!("{state} host depth bounds {depth_bounds}");
            let made = mesh_pair_pipeline(ms, ps, &made_of, |pair| {
                let (ms_code, ps_code) = match pair {
                    Some(pair) => (bytecode(&pair.0.dxil), bytecode(&pair.1.dxil)),
                    None => (code(0), code(1)),
                };
                // "triangles": the stream names its topology since it did not
                // load from the library without one (see `mesh_pipeline`); a
                // new name keeps such entries, stored before, from being
                // looked up again.
                let name = super::library::name(
                    "mesh triangles",
                    &root_blob,
                    &[&code_bytes(&ms_code), &code_bytes(&ps_code)],
                    &state,
                );
                let made = mesh_pipeline(&host.device, &root, ms_code, ps_code, g, depth_bounds, &name);
                if pair.is_some() && made.is_err() {
                    // What the host said of the linked pair, before the shared one is made.
                    host.log_messages();
                }
                made
            });
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    // The state too: the shaders alone make a pipeline
                    // with another state (`victory translate` pairs).
                    tracing::warn!(error = %e, ?shaders, state = %state, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
                #[cfg(feature = "diagnostics")]
                program: std::sync::OnceLock::new(),
            });
        }

        // A pipeline the title makes for rectangles gets a geometry stage
        // that completes each one from the three vertices the vertex
        // shader made. Without it (refused here or by the host) the draws
        // fall back on what `draw` does for rectangles.
        let mut rect_gs = None;
        if g.topology.0 == RECTANGLES {
            let ps = stages.get(1).and_then(|(s, _)| s.translated());
            let vs = &stages[0].0.translated()?.host.dxil;
            match victory_shader::emit::rect_list_gs(vs, ps.map(|t| &t.host.dxil[..])) {
                Ok(gs) => rect_gs = Some(gs),
                Err(e) => tracing::warn!(error = %e, ?shaders, "no rectangle geometry stage"),
            }
        }
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: rect_gs.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |gs| {
                D3D12_SHADER_BYTECODE {
                    pShaderBytecode: gs.as_ptr().cast(),
                    BytecodeLength: gs.len(),
                }
            }),
            BlendState: host_blend(&g.blend, &g.targets),
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        // A pipeline with the geometry stage has a name of its own in the
        // library; the others keep theirs.
        let name = |gs: Option<&Vec<u8>>| match gs {
            Some(gs) => super::library::name(
                "graphics rectangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs[..]],
                &state,
            ),
            None => super::library::name(
                "graphics",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            ),
        };
        let make = |desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, name: &str| {
            if depth_bounds {
                return graphics_depth_bounds_pipeline(&host.device, &root, desc, g.depth1(), name);
            }
            super::library::pipeline(
                name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(desc) },
            )
        };
        let mut made = make(&desc, &name(rect_gs.as_ref()));
        if made.is_err() && rect_gs.is_some() {
            // The host refused the geometry stage: once more without it.
            tracing::warn!(?shaders, "host pipeline refused with the rectangle geometry stage");
            host.log_messages();
            desc.GS = D3D12_SHADER_BYTECODE::default();
            rect_gs = None;
            made = make(&desc, &name(None));
        }
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, state = %state, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            depth_bounds,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            rects: rect_gs.is_some(),
            #[cfg(feature = "diagnostics")]
            program: std::sync::OnceLock::new(),
            state: {
                let (d, r, b) = (&g.depth, &g.rasterizer, &g.blend.RenderTarget[0]);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC| {
                    format!(
                        "fail {} depth-fail {} pass {} func {}",
                        f.StencilFailOp.0, f.StencilDepthFailOp.0, f.StencilPassOp.0, f.StencilFunc.0
                    )
                };
                let bounds = if depth_bounds { " bounds true" } else { "" };
                format!(
                    "depth {} write {} func {}{bounds}; stencil {} read {:#x} write {:#x} front [{}] back [{}]; cull {} front-ccw {}; blend {} src {} dst {} op {} mask {:#x}",
                    d.DepthEnable.as_bool(),
                    d.DepthWriteMask.0,
                    d.DepthFunc.0,
                    d.StencilEnable.as_bool(),
                    d.StencilReadMask,
                    d.StencilWriteMask,
                    face(&d.FrontFace),
                    face(&d.BackFace),
                    r.CullMode.0,
                    r.FrontCounterClockwise.as_bool(),
                    b.BlendEnable.as_bool(),
                    b.SrcBlend.0,
                    b.DestBlend.0,
                    b.BlendOp.0,
                    b.RenderTargetWriteMask
                )
            },
        })
    }
}

/// A stream subobject: its type, then its description, pointer-aligned.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

fn stream_pipeline<S>(
    device: &ID3D12Device,
    stream: &mut S,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<S>(),
        pPipelineStateSubobjectStream: std::ptr::from_mut(stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// A graphics pipeline enabling the depth bounds test: `D3D12_GRAPHICS_PIPELINE_STATE_DESC`
/// has no field for `DepthBoundsTestEnable`, so it is created and loaded as a
/// pipeline-state stream with `DEPTH_STENCIL1`.
fn graphics_depth_bounds_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC1,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        vs: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC1>,
        input: Sub<D3D12_INPUT_LAYOUT_DESC>,
        strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    #[repr(C)]
    struct StreamGs {
        base: Stream,
        gs: Sub<D3D12_SHADER_BYTECODE>,
    }
    let targets = D3D12_RT_FORMAT_ARRAY {
        RTFormats: d.RTVFormats,
        NumRenderTargets: d.NumRenderTargets,
    };
    let base = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, d.VS),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, d.PS),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, d.BlendState),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, d.SampleMask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, d.RasterizerState),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
        input: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, d.InputLayout),
        strip_cut: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, d.IBStripCutValue),
        topology: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, d.PrimitiveTopologyType),
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, d.DSVFormat),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, d.SampleDesc),
    };
    if d.GS.pShaderBytecode.is_null() {
        let mut stream = base;
        stream_pipeline(device, &mut stream, name)
    } else {
        let mut stream = StreamGs {
            base,
            gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, d.GS),
        };
        stream_pipeline(device, &mut stream, name)
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout).
///
/// The stream names a triangle topology, which D3D12 does not need for a
/// mesh shader: without it the pipeline library refuses to load a stored
/// pipeline with the very stream it was made from (`E_INVALIDARG`; debug
/// layer message 972, "does not match the one used to create the PSO"), so
/// every mesh pipeline was compiled again, and its store then refused (970,
/// "already exists"). The translation makes triangle mesh shaders only
/// (`mesh_prologue` refuses others), and the debug layer wants the two to
/// agree (1323).
/// The mesh program `ms` and the pixel program `ps` translated for one pipeline
/// ([`victory_shader::emit::link_mesh`]), when `ms`'s outputs pass D3D12's 32 KiB as it is
/// translated alone: a console mesh program can declare more output components than
/// that holds at its vertex maximum, which the validator refuses; NVIDIA's driver
/// built such programs anyway. Linked,
/// the mesh program writes only the outputs `ps` reads and `ps` declares only the inputs it
/// reads. Made once per pair (both programs translated again, not kept on disk: the pipeline
/// library keeps the pipeline); `None` when `ms` fits or the pair does not link: then the
/// pipeline is made of the shared translations, as before. `VICTORY_MESH_LINK=0` turns it
/// off.
/// Comes with what names the pair ([`pair_key`]).
fn linked_as(ms: &Arc<Shader>, ps: &Arc<Shader>) -> Option<(PairKey, Arc<LinkedPair>)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| link_on(std::env::var_os("VICTORY_MESH_LINK").as_deref())) {
        return None;
    }
    let (tm, tp) = (ms.translated()?, ps.translated()?);
    if victory_shader::emit::mesh_fits(&tm.host.dxil).unwrap_or(true) {
        return None;
    }
    let key = pair_key(ms, ps);
    if let Some(pair) = PAIRS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .get(&key)
    {
        return pair.clone().map(|pair| (key, pair));
    }
    // (translated outside the lock: two workers on one pair translate it twice at worst, and
    // both get what the first of them published)
    let pair = match victory_shader::emit::link_mesh(&ms.container, &tm.host, &ps.container, &tp.host) {
        Ok(Some(pair)) => {
            tracing::info!(shaders = ?[ms, ps], "mesh outputs over 32 KiB: the pair is linked for its pipeline");
            Some(Arc::new(pair))
        }
        Ok(None) => {
            // Why, when it is the interface between the two (`mesh_interface`).
            match victory_shader::emit::mesh_interface(&tm.host, &ps.container, &tp.host) {
                Err(e) => {
                    tracing::warn!(shaders = ?[ms, ps], error = %e, "mesh outputs over 32 KiB, and the pair is not linked");
                }
                Ok(()) => {
                    tracing::warn!(shaders = ?[ms, ps], "mesh outputs over 32 KiB, and the pair does not link");
                }
            }
            None
        }
        Err(e) => {
            tracing::warn!(shaders = ?[ms, ps], error = %e, "mesh outputs over 32 KiB, and the pair does not translate linked");
            None
        }
    };
    publish(key, pair).map(|pair| (key, pair))
}

/// Tests: the pair of [`linked_as`].
#[cfg(test)]
fn linked(ms: &Arc<Shader>, ps: &Arc<Shader>) -> Option<Arc<LinkedPair>> {
    linked_as(ms, ps).map(|(_, pair)| pair)
}

/// Puts what [`linked_as`] found for a pair into [`PAIRS`], unless the pair has an entry by
/// now (another worker translated it meanwhile): what the pair has there, which is the
/// first publication for good.
fn publish(key: PairKey, pair: Pair) -> Pair {
    PAIRS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .entry(key)
        .or_insert(pair)
        .clone()
}

/// Whether pairs are linked, by what `VICTORY_MESH_LINK` holds: not when it is exactly `0`.
/// Off, [`linked_as`] has no pair for anyone before it looks at a program, so every mesh
/// pipeline is made of the shared translations and of nothing else.
fn link_on(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// A mesh program and a pixel program translated for one pipeline ([`linked_as`]).
type LinkedPair = (victory_shader::emit::HostShader, victory_shader::emit::HostShader);

/// What [`linked_as`] has for a mesh and a pixel program: `None` for a pair that does not link.
type Pair = Option<Arc<LinkedPair>>;

/// What names a pair in [`PAIRS`]: the whole digests of the two containers.
type PairKey = ([u8; 32], [u8; 32]);

/// Every pair [`linked_as`] was asked for.
static PAIRS: std::sync::Mutex<Option<std::collections::HashMap<PairKey, Pair>>> =
    std::sync::Mutex::new(None);

/// What names the pair of `ms` and `ps` in [`PAIRS`]. Not the ids the logs print
/// ([`Shader::hash`]): those are the first 64 bits of the digests, and two pixel programs
/// that share them would share a pair, the second drawn with a mesh program pruned for the
/// first. Hashed here, where only a mesh program over the limit comes.
fn pair_key(ms: &Shader, ps: &Shader) -> PairKey {
    (
        victory_shader::container_id(&ms.container).0,
        victory_shader::container_id(&ps.container).0,
    )
}

/// The linked pipelines the host refused ([`mesh_pair_pipeline`]): the pair, and everything
/// else the pipeline was made of. Kept while the process lives, as the host device is.
static REFUSED: std::sync::Mutex<Option<std::collections::HashSet<(PairKey, String)>>> =
    std::sync::Mutex::new(None);

/// Whether a failure to make a pipeline is the host's refusal of what the pipeline was
/// described as, which it would give again: `E_INVALIDARG`, and no other result. Every other
/// one is taken to say nothing of the description (a generic `E_FAIL`, a driver's internal
/// error, a lost device, no memory, an interface the object does not have, a result nobody
/// here has seen): asked again, the same pipeline may be made.
///
/// The list is of what is known to be a refusal, not of what is known not to be: a result
/// wrongly left out costs one more attempt per pipeline of the pair and state, one wrongly
/// put in costs the linked pair for the life of the process.
fn refuses(e: &windows::core::Error) -> bool {
    e.code() == windows::Win32::Foundation::E_INVALIDARG
}

/// What `make` makes for the mesh program `ms` drawn with the pixel program `ps`: of their
/// linked pair when they have one ([`linked_as`]; `make` is handed it), else of the shared
/// translations (`make(None)`). `state` is everything else the pipeline is made of.
///
/// A driver can take a mesh program whose outputs pass 32 KiB (NVIDIA's does) and still
/// refuse the linked pair made in its place. The pipeline is then made of the shared
/// translations after all, as it was before pairs were linked, rather than not at all (its
/// draws would be dropped).
///
/// What was refused is that pipeline: the linked pair in `state`. It is not asked for again
/// ([`REFUSED`]), and a warning says so once; the same pair in another state is still asked
/// for linked (a state the host does not take says nothing of the pair). Only a failure
/// that is a refusal of the description ([`refuses`]) is remembered; after any other this
/// pipeline is made of the shared translations, and the next one of the pair and state asks
/// for the linked pair again.
fn mesh_pair_pipeline<T>(
    ms: &Arc<Shader>,
    ps: Option<&Arc<Shader>>,
    state: &str,
    make: impl Fn(Option<&LinkedPair>) -> windows::core::Result<T>,
) -> windows::core::Result<T> {
    let refused = |refusal: &(PairKey, String)| {
        REFUSED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|refused| refused.contains(refusal))
    };
    if let Some(ps) = ps
        && let Some((key, pair)) = linked_as(ms, ps)
    {
        let refusal = (key, state.to_owned());
        if !refused(&refusal) {
            match make(Some(&pair)) {
                Ok(made) => return Ok(made),
                Err(e) if !refuses(&e) => {
                    tracing::warn!(
                        error = %e,
                        shaders = ?[ms, ps],
                        "the pipeline of the linked pair was not made: this one is made of the shared translations"
                    );
                }
                Err(e) => {
                    let first = REFUSED
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_or_insert_with(Default::default)
                        .insert(refusal);
                    if first {
                        tracing::warn!(
                            error = %e,
                            shaders = ?[ms, ps],
                            state,
                            "the host refused the pipeline of the linked pair: in this state the pair's pipelines are made of the shared translations from here on"
                        );
                    }
                }
            }
        }
    }
    make(None)
}

fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    depth_bounds: bool,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    #[repr(C)]
    struct Stream<D> {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    // The title's blend state as the host takes it (see `host_blend`).
    let blend = host_blend(&g.blend, &g.targets);
    let build = |depth: Sub<_>| {
        let mut stream = Stream {
            root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
            ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
            raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
            depth,
            targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
            dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
            samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
            topology: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
        };
        stream_pipeline(device, &mut stream, name)
    };
    if depth_bounds {
        let mut stream = Stream {
            root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
            ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
            raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
            depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, g.depth1()),
            targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
            dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
            samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
            topology: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
        };
        stream_pipeline(device, &mut stream, name)
    } else {
        build(Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth))
    }
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }.to_vec()
}

/// Whether static samplers may be anisotropic with a point mip filter on
/// the host device: the device says so (`D3D12_OPTIONS19`), and a root
/// signature with such a sampler serializes and is made on it. Asked once,
/// and logged. (The runtime's serializer refuses the filter as
/// unrecognized on a software adapter that says it supports it; without
/// either, the linear mip filter the translation had before is used.)
fn aniso_point_mip(host: &crate::host::Host) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        // SAFETY: writes `options` of the given size; a runtime that does
        // not know the query refuses it, which means no support.
        let said = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.AnisoFilterWithPointMipSupported.as_bool();
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_ANISOTROPIC_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MaxAnisotropy: 16,
            ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            MaxLOD: D3D12_FLOAT32_MAX,
            RegisterSpace: SAMPLER_SPACE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            ..Default::default()
        };
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumStaticSamplers: 1,
                    pStaticSamplers: &sampler,
                    ..Default::default()
                },
            },
        };
        // SAFETY: the description points at a live sampler; the blob is
        // read before it is released.
        let made = said
            && unsafe {
                let mut blob: Option<ID3DBlob> = None;
                D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).is_ok()
                    && blob.is_some_and(|blob| {
                        let bytes =
                            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
                        host.device.CreateRootSignature::<ID3D12RootSignature>(0, bytes).is_ok()
                    })
            };
        tracing::info!(
            device = said,
            root_signature = made,
            "host static samplers anisotropic with a point mip filter (else with the linear one)"
        );
        made
    })
}

/// The host static sampler of a translated stage's immediate sampler `s`,
/// at register `register` of the sampler space: of the filter the host
/// device takes ([`aniso_point_mip`]).
fn static_sampler(
    host: &crate::host::Host,
    s: &victory_shader::emit::StaticSampler,
    register: u32,
    visibility: D3D12_SHADER_VISIBILITY,
) -> D3D12_STATIC_SAMPLER_DESC {
    let point_mip = s.filter & !0x180 == 0x54 && aniso_point_mip(host);
    D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER(s.filter_for_host(point_mip) as i32),
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
        MipLODBias: s.mip_lod_bias,
        MaxAnisotropy: s.max_anisotropy,
        ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
        BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
        MinLOD: s.min_lod,
        MaxLOD: s.max_lod,
        ShaderRegister: register,
        RegisterSpace: SAMPLER_SPACE,
        ShaderVisibility: visibility,
    }
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
impl crate::host::Host {
    /// Whether this device is on the software adapter (WARP or the basic render driver): asked of the adapter the
    /// device reports, not of a switch. Used by the tests that need a card.
    pub(crate) fn is_software(&self) -> bool {
        victory_shader::adapter::device_adapter(&self.device).is_some_and(|adapter| adapter.software)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    /// An immediate sampler that is anisotropic with a point mip filter
    /// (filtering, minimum and maximum reduction) makes a static sampler
    /// that this device's root signature takes, whether or not the device
    /// and the runtime's serializer take that filter: the reduction and the
    /// LOD bounds stay the translation's.
    #[test]
    fn a_point_mip_anisotropic_static_sampler_makes_a_root_signature_on_this_device() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        for reduction in 0..3u32 {
            // Anisotropic magnification and minification, point mip filter,
            // LOD 1 to 5.25.
            let s = victory_shader::emit::StaticSampler::from_words([
                reduction << 29,
                0x0054_0100,
                1 << 26 | 0x00F0_0000,
                0,
            ]);
            assert_eq!(s.filter & !0x180, 0x54);
            let desc = static_sampler(host, &s, 0, D3D12_SHADER_VISIBILITY_ALL);
            if let Err(e) = root_signature(&host.device, &[], &[desc], true) {
                panic!("reduction {reduction}, filter {:#x}: {e}", desc.Filter.0);
            }
            assert_eq!(desc.Filter.0 as u32 & 0x180, s.filter & 0x180, "the reduction");
            assert_eq!((desc.MinLOD, desc.MaxLOD), (1.0, 5.25));
        }
    }

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shader support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        // This pipeline has only a color target, with no depth buffer.
        g.depth.DepthEnable = false.into();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, false, &name).expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, false, &name).expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }

    /// A mesh shader covering the whole target, and a pixel shader writing
    /// an integer target first, then two colour targets.
    const MS_COVER: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(-1, -1, 0, 1);
            v[1].p = float4(-1, 3, 0, 1);
            v[2].p = float4(3, -1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS_THREE: &str = "struct O { uint a : SV_Target0; float4 b : SV_Target1; float4 c : SV_Target2; };
        O main() { O o; o.a = 4321; o.b = float4(0, 1, 0, 0.5); o.c = float4(1, 0, 0, 0.5); return o; }";

    /// A mesh pipeline with blending on in one description shared by every
    /// target (independent blending off), the first target R16_UINT and so
    /// the shared description's own slot. The host refuses blending into an
    /// integer format; the pipeline is made all the same, and its draw
    /// blends the two colour targets and writes the integer one as the
    /// shader says.
    #[test]
    fn a_mesh_pipeline_blending_into_an_integer_target_is_made_and_draws() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shader support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS_COVER, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS_THREE, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let formats = [
            DXGI_FORMAT_R16_UINT,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8G8B8A8_UNORM,
        ];
        let mut g = GraphicsState::default();
        g.depth.DepthEnable = false.into();
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.targets.NumRenderTargets = 3;
        g.targets.RTFormats[..3].copy_from_slice(&formats);
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.blend.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        g.blend.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        let name = super::super::library::name("mesh integer blend test", &root_blob, &[&ms, &ps], &g.describe());
        let pso = mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, false, &name)
            .expect("a mesh pipeline blending into an R16_UINT target made");

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let targets: Vec<ID3D12Resource> = formats
            .iter()
            .map(|&format| {
                let rd = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 4,
                    Height: 4,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let mut out: Option<ID3D12Resource> = None;
                // SAFETY: valid heap and resource descriptions.
                unsafe {
                    host.device
                        .CreateCommittedResource(
                            &heap,
                            D3D12_HEAP_FLAG_NONE,
                            &rd,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            None,
                            &mut out,
                        )
                        .unwrap();
                }
                out.unwrap()
            })
            .collect();
        let readback = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 3 * 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid readback buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &rd,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };

        let list = host.new_list().unwrap();
        let rtvs: Vec<_> = targets.iter().map(|t| host.rtv(Some(t), None)).collect();
        let l6 = windows::core::Interface::cast::<ID3D12GraphicsCommandList6>(&list.list).unwrap();
        // SAFETY: an open command list, live resources and views.
        unsafe {
            // What the targets hold before: 7; and blue, opaque.
            list.list.ClearRenderTargetView(rtvs[0], &[7.0; 4], None);
            list.list.ClearRenderTargetView(rtvs[1], &[0.0, 0.0, 1.0, 1.0], None);
            list.list.ClearRenderTargetView(rtvs[2], &[0.0, 0.0, 1.0, 1.0], None);
            list.list.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, None);
            list.list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            }]);
            list.list.SetGraphicsRootSignature(&root);
            list.list.SetPipelineState(&pso);
            l6.DispatchMesh(1, 1, 1);
            for (i, (texture, &format)) in targets.iter().zip(&formats).enumerate() {
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: i as u64 * 1024,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: format,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
        }
        list.close();
        host.execute(&[&list]);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `execute` waited for the GPU; the buffer holds 3 KiB.
        let bytes = unsafe {
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), 3 * 1024).to_vec();
            readback.Unmap(0, None);
            bytes
        };
        // R16_UINT: the shader's value, unblended.
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 4321, "the R16_UINT target holds the shader's value");
        // RGBA16F: green at half alpha over blue, every channel exactly 0.5
        // but red.
        let half = |i: usize| u16::from_le_bytes([bytes[1024 + 2 * i], bytes[1024 + 2 * i + 1]]);
        let rgba16 = [half(0), half(1), half(2), half(3)];
        assert_eq!(rgba16, [0, 0x3800, 0x3800, 0x3800], "the RGBA16F target is blended");
        // RGBA8: red at half alpha over blue is half red, half blue (not the
        // shader's red alone).
        let rgba8 = &bytes[2048..2052];
        assert!(
            (126..=129).contains(&rgba8[0]) && rgba8[1] == 0 && (126..=129).contains(&rgba8[2]),
            "the RGBA8 target is blended: {rgba8:?}"
        );
    }

    const VS: &str = "float4 main(uint i : SV_VertexID) : SV_Position {
        float2 p = float2((i << 1) & 2, i & 2);
        return float4(p * 2.0 - 1.0, 0.5, 1.0);
    }";

    /// Why the depth-bounds case is not run, if it is not. The software
    /// adapter says it takes the test and draws through it all the same
    /// (depth 0.5 passes bounds of [0.6, 1.0]): the case is for a card. A run
    /// that asked for the software adapter (`asked_software`) skips it, as
    /// before. A runner whose only device is software skips it when the
    /// CI-only variable asks (`allow_skip`); a desk that did not ask for the
    /// software adapter and has no card still fails, as it should.
    fn depth_bounds_skip(asked_software: bool, software: bool, allow_skip: bool) -> Option<&'static str> {
        if asked_software {
            Some("the software adapter does not apply depth bounds; skipped")
        } else if software && allow_skip {
            Some("the software adapter does not apply depth bounds; skipped (VICTORY_TESTS_ALLOW_SOFTWARE_SKIP=1)")
        } else {
            None
        }
    }

    /// The skip's three cases without a card-less runner: the CI variable's
    /// branch is reached only where the device is software and no one asked
    /// for it, which a desk with `VICTORY_D3D12_WARP=1` never shows.
    #[test]
    fn the_depth_bounds_case_skips_the_software_adapter_only_when_asked_or_on_ci() {
        assert_eq!(
            depth_bounds_skip(false, true, true),
            Some("the software adapter does not apply depth bounds; skipped (VICTORY_TESTS_ALLOW_SOFTWARE_SKIP=1)"),
            "a card-less runner with the variable"
        );
        assert_eq!(depth_bounds_skip(false, true, false), None, "a desk on the software adapter that did not ask: fails loudly");
        assert_eq!(depth_bounds_skip(false, false, true), None, "a card runs it, the variable or not");
        assert_eq!(
            depth_bounds_skip(true, true, false),
            Some("the software adapter does not apply depth bounds; skipped"),
            "a run that asked for the software adapter"
        );
    }

    /// A graphics pipeline with `DepthBoundsTestEnable` loads back from the
    /// pipeline library and culls or passes fragments according to
    /// `OMSetDepthBounds` (including across a command-list cut).
    #[test]
    fn depth_bounds_pipeline_loads_and_filters_draws() {
        let Some(host) = crate::host::test_host() else {
            return;
        };
        if !host.depth_bounds() {
            eprintln!("no host depth bounds support; skipped");
            return;
        }
        if let Some(reason) = depth_bounds_skip(
            crate::host::on_warp(),
            host.is_software(),
            std::env::var("VICTORY_TESTS_ALLOW_SOFTWARE_SKIP").is_ok_and(|v| v == "1"),
        ) {
            eprintln!("{reason}");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let vs = compile(&dxc, dir.path(), VS, "vs_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();

        let mut g = GraphicsState::default();
        // Subobject 21 (`DEPTH_STENCIL1`) with `DepthBoundsTestEnable = true`,
        // depth test itself off so only depth bounds gates the draw.
        #[repr(C, align(8))]
        struct Sub21(u32, D3D12_DEPTH_STENCIL_DESC1);
        let sub = Sub21(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: `sub` is a complete subobject 21.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds);
        assert!(g.describe().contains("depth-bounds"));
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;

        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed pointer, never released by the descriptor.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(&vs),
            PS: code(&ps),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            IBStripCutValue: g.strip_cut,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::super::library::name(
            "graphics depth-bounds test",
            &root_blob,
            &[&vs, &ps],
            &g.describe(),
        );
        let pso = graphics_depth_bounds_pipeline(&host.device, &root, &desc, g.depth1(), &name)
            .expect("depth-bounds PSO made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        graphics_depth_bounds_pipeline(&host.device, &root, &desc, g.depth1(), &name)
            .expect("depth-bounds PSO made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the depth-bounds pipeline was compiled again instead of loaded from the library"
        );

        // Execute two draws against a depth buffer cleared to 0.5:
        // 1) bounds [0.6, 1.0] -> rejected (target stays 0)
        // 2) bounds [0.4, 0.6] replayed onto a fresh HostList -> passes (target becomes 255)
        let make_tex = |format, flags, state, clear: D3D12_CLEAR_VALUE| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource descriptions.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, state, Some(&clear), &mut out)
                    .unwrap();
            }
            out.unwrap()
        };
        let color = make_tex(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
            },
        );
        let depth = make_tex(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 0.5,
                        Stencil: 0,
                    },
                },
            },
        );
        let readback = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid readback buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &rd,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };

        let run_draw = |min: f32, max: f32| -> u8 {
            let mut state = crate::draw::State::default();
            let first_list = host.new_list().unwrap();
            state.set_depth_bounds(&first_list, min, max);
            first_list.close();
            // Replay onto a second host list, exercising `State::replay`.
            let list = host.new_list().unwrap();
            state.replay(&list);
            let rtv = host.rtv(Some(&color), None);
            let dsv = host.dsv(&depth, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let rect = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            };
            // SAFETY: open command list, live resources and views.
            unsafe {
                list.list.ClearRenderTargetView(rtv, &[0.0; 4], None);
                list.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
                list.list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                list.list.RSSetViewports(&[vp]);
                list.list.RSSetScissorRects(&[rect]);
                list.list.SetGraphicsRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list
                    .IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list.list.DrawInstanced(3, 1, 0, 0);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&color),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: `execute` waited for the GPU; readback holds 1024 bytes.
            unsafe {
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let pixel = *ptr.cast::<u8>();
                readback.Unmap(0, None);
                pixel
            }
        };

        assert_eq!(run_draw(0.6, 1.0), 0, "depth 0.5 outside [0.6, 1.0] must be rejected");
        assert_eq!(run_draw(0.4, 0.6), 255, "depth 0.5 inside [0.4, 0.6] must pass");
    }

    fn targets(formats: &[DXGI_FORMAT]) -> D3D12_RT_FORMAT_ARRAY {
        let mut t = D3D12_RT_FORMAT_ARRAY {
            NumRenderTargets: formats.len() as u32,
            ..Default::default()
        };
        t.RTFormats[..formats.len()].copy_from_slice(formats);
        t
    }

    fn blending(independent: bool) -> D3D12_BLEND_DESC {
        let mut b = GraphicsState::default().blend;
        b.IndependentBlendEnable = independent.into();
        for t in &mut b.RenderTarget {
            t.BlendEnable = true.into();
            t.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            t.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        }
        b
    }

    #[test]
    fn integer_targets_do_not_blend() {
        let t = targets(&[
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8_UINT,
        ]);
        for independent in [false, true] {
            let host = host_blend(&blending(independent), &t);
            assert!(host.IndependentBlendEnable.as_bool());
            assert!(host.RenderTarget[0].BlendEnable.as_bool());
            assert!(host.RenderTarget[1].BlendEnable.as_bool());
            assert!(!host.RenderTarget[2].BlendEnable.as_bool());
            // The rest of the target's state is the title's.
            assert_eq!(host.RenderTarget[2].SrcBlend, D3D12_BLEND_SRC_ALPHA);
            assert_eq!(host.RenderTarget[0].DestBlend, D3D12_BLEND_INV_SRC_ALPHA);
        }
    }

    #[test]
    fn blend_states_without_integer_blending_are_kept() {
        // No integer target.
        let b = blending(false);
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT]);
        assert_eq!(host_blend(&b, &t), b);
        // An integer target that does not blend.
        let mut b = blending(true);
        b.RenderTarget[1].BlendEnable = false.into();
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT]);
        assert_eq!(host_blend(&b, &t), b);
        // Integer formats past the targets in use.
        let mut t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM]);
        t.RTFormats[1] = DXGI_FORMAT_R8_UINT;
        assert_eq!(host_blend(&blending(false), &t), blending(false));
    }
}

#[cfg(test)]
mod blend_tests {
    use super::*;

    fn state(independent: bool, formats: &[DXGI_FORMAT]) -> GraphicsState {
        let mut g = GraphicsState::default();
        g.blend.IndependentBlendEnable = independent.into();
        let max = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ONE,
            BlendOp: D3D12_BLEND_OP_MAX,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: 15,
        };
        g.blend.RenderTarget = [max; 8];
        g.targets.NumRenderTargets = formats.len() as u32;
        g.targets.RTFormats[..formats.len()].copy_from_slice(formats);
        g
    }

    /// The windshield wiper mask: MAX into R32_FLOAT, B4G4R4A4 and R8_UINT.
    #[test]
    fn integer_targets_take_no_blending() {
        let not_uint = |f: DXGI_FORMAT| f != DXGI_FORMAT_R8_UINT;
        let formats = [DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8_UINT];
        for independent in [true, false] {
            let b = blend_where(&state(independent, &formats), not_uint);
            let on: Vec<bool> = b.RenderTarget[..3].iter().map(|t| t.BlendEnable.as_bool()).collect();
            assert_eq!(on, [true, true, false], "independent {independent}");
            assert!(b.IndependentBlendEnable.as_bool());
            assert_eq!(b.RenderTarget[1].BlendOp, D3D12_BLEND_OP_MAX, "the others keep their blending");
        }
        // Nothing to change: the title's state as it is.
        let g = state(false, &[DXGI_FORMAT_R32_FLOAT]);
        assert!(!blend_where(&g, not_uint).IndependentBlendEnable.as_bool());
    }
}
