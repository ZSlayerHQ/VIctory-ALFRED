//! Draws a straight-alpha canvas from its existing card buffer on the present list.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use super::{Buffers, Host, HostList, transition};

const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0, visibility=SHADER_VISIBILITY_PIXEL), SRV(t0, visibility=SHADER_VISIBILITY_PIXEL)"
cbuffer Canvas : register(b0) { uint2 origin; uint pitch; uint halfPixels; };
ByteAddressBuffer pixels : register(t0);

float4 vertex(uint i : SV_VertexID) : SV_Position {
    return float4(i == 1 ? 3.0 : -1.0, i == 2 ? -3.0 : 1.0, 0.0, 1.0);
}

[RootSignature(RS)]
float4 pixel(float4 position : SV_Position) : SV_Target {
    uint2 xy = uint2(position.xy) - origin;
    uint row = xy.y * pitch;
    float4 colour;
    if (halfPixels) {
        uint2 p = pixels.Load2(row + xy.x * 8);
        colour = f16tof32(uint4(p.x & 65535, p.x >> 16, p.y & 65535, p.y >> 16));
    } else {
        uint p = pixels.Load(row + xy.x * 4);
        colour = float4((p >> uint4(0, 8, 16, 24)) & 255) / 255.0;
    }
    if (colour.a == 0.0) discard;
    return colour;
}
"#;

/// Made once per swap-chain format, only when RGBA art is first used.
#[derive(Default)]
pub(in crate::host) struct Blend {
    pipeline: Option<Pipeline>,
    attempted: bool,
}

impl Blend {
    pub(in crate::host) fn ready(&mut self, host: &Host, format: DXGI_FORMAT) -> bool {
        if !std::mem::replace(&mut self.attempted, true) {
            match Pipeline::new(host, format) {
                Ok(pipeline) => self.pipeline = Some(pipeline),
                Err(error) => tracing::warn!(%error, format = format.0, "could not make the overlay blend pipeline"),
            }
        }
        self.pipeline.is_some()
    }

    pub(super) fn pipeline(&self) -> Option<&Pipeline> {
        self.pipeline.as_ref()
    }
}

pub(super) struct Pipeline {
    root: ID3D12RootSignature,
    state: ID3D12PipelineState,
}

impl Pipeline {
    fn new(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let vs = crate::host::compile(SHADER, s!("vertex"), s!("vs_5_1"))?;
        let ps = crate::host::compile(SHADER, s!("pixel"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: shader data and the borrowed root signature live through creation.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &ps)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_SRC_ALPHA,
                DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let state = host.device.CreateGraphicsPipelineState(&desc)?;
            Ok(Self { root, state })
        }
    }

    pub(super) fn draw(
        &self,
        host: &Host,
        list: &HostList,
        buffers: &mut Buffers,
        back: &ID3D12Resource,
        part: (u32, u32),
        at: (u32, u32),
    ) {
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let rtv = host.rtv(Some(back), None);
        let list = &list.list;
        let constants = [
            at.0,
            at.1,
            buffers.pitch,
            u32::from(buffers.format == DXGI_FORMAT_R16G16B16A16_FLOAT),
        ];
        let viewport = D3D12_VIEWPORT {
            TopLeftX: at.0 as f32,
            TopLeftY: at.1 as f32,
            Width: part.0 as f32,
            Height: part.1 as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: at.0 as i32,
            top: at.1 as i32,
            right: (at.0 + part.0) as i32,
            bottom: (at.1 + part.1) as i32,
        };
        // SAFETY: the list is open, the rectangle is inside both resources, and the
        // presenter and its retirement queue hold the resources through submission.
        unsafe {
            let source = if std::mem::take(&mut buffers.fresh) {
                list.CopyResource(&buffers.card, &buffers.uploads[buffers.turn].0);
                D3D12_RESOURCE_STATE_COPY_DEST
            } else {
                D3D12_RESOURCE_STATE_COMMON
            };
            list.ResourceBarrier(&[
                transition(&buffers.card, source, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                transition(back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
            ]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.state);
            list.SetGraphicsRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.SetGraphicsRootShaderResourceView(1, buffers.card.GetGPUVirtualAddress());
            list.RSSetViewports(&[viewport]);
            list.RSSetScissorRects(&[scissor]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
            list.ResourceBarrier(&[
                transition(back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(
                    &buffers.card,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
            ]);
        }
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests;
