//! The lowering's linked mode ([`super::super::link::with_linked`]): a stage in a host
//! pipeline with a partner declared only the signature elements the partner uses. The
//! elements stay in the signature while the program is lowered (rows of 16-bit values and the
//! like are found by the elements around them); the stores to an output left out are not
//! made, and what is left out goes at the end, its ids counted again.

use super::super::container::SigElement;
use super::{Lower, Stop};
use crate::ShaderError;
use crate::emit::link;

impl Lower<'_> {
    /// Whether this stage leaves each input and output out ([`link::left_out`]).
    fn dropped(&self) -> (Vec<bool>, Vec<bool>) {
        link::left_out(&self.s.inputs, &self.s.outputs)
    }

    /// The DXIL signature id of element `id` of the console declaration: ids count only the
    /// elements kept (an input left out is never read, an output left out never stored).
    pub(super) fn sig_id(&self, input: bool, id: usize) -> u32 {
        let (drop_in, drop_out) = self.dropped();
        let drop = if input { &drop_in } else { &drop_out };
        (id - drop.iter().take(id).filter(|&&d| d).count()) as u32
    }

    /// Whether the store to vertex output `id` (or to a primitive output) is not made.
    pub(super) fn dropped_output(&self, id: usize, primitive: bool) -> bool {
        !primitive && self.dropped().1.get(id).copied().unwrap_or(false)
    }

    /// The element id a store names in the emitted signature.
    pub(super) fn store_id(&self, primitive: bool, id: usize) -> u32 {
        if primitive {
            id as u32
        } else {
            self.sig_id(false, id)
        }
    }

    /// Whether a mesh program's outputs fit D3D12's 32 KiB with the outputs a linked pipeline
    /// declares: the pixel program reads only some of them there.
    pub(super) fn mesh_outputs_fit(&self, mesh: super::super::shader::Mesh) -> bool {
        use super::super::container::{MESH_OUTPUT_LIMIT, mesh_output_bytes};
        let (_, drop) = self.dropped();
        let outputs: Vec<SigElement> = if drop.is_empty() {
            self.s.outputs.clone()
        } else {
            self.s
                .outputs
                .iter()
                .zip(&drop)
                .filter(|&(_, &d)| !d)
                .map(|(e, _)| e.clone())
                .collect()
        };
        mesh_output_bytes(
            &outputs,
            &self.s.prims,
            mesh.max_vertices,
            mesh.max_primitives,
        ) <= MESH_OUTPUT_LIMIT
    }

    /// The linked mode at the end of lowering: an input left out must be one the program
    /// never read, then the signatures lose the elements left out.
    pub(super) fn linked(&mut self) -> Result<(), Stop> {
        let (drop_in, drop_out) = self.dropped();
        if drop_in.is_empty() && drop_out.is_empty() {
            return Ok(());
        }
        if self
            .s
            .inputs
            .iter()
            .zip(&drop_in)
            .any(|(e, &d)| d && e.usage != 0)
        {
            return Err(Stop::Fail(ShaderError::Unsupported {
                offset: 0,
                what: "an input left out of a linked pipeline read",
            }));
        }
        let keep = |els: &mut Vec<SigElement>, drop: &[bool]| {
            let mut d = drop.iter();
            els.retain(|_| !d.next().copied().unwrap_or(false));
        };
        keep(&mut self.s.inputs, &drop_in);
        keep(&mut self.s.outputs, &drop_out);
        Ok(())
    }
}
