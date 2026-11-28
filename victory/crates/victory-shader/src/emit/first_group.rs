//! The number of a compute program's first thread group, for a dispatch the host runs in
//! parts (`VICTORY_SPLIT_DISPATCH=1`, off by default: compute
//! programs add the number of the first group).
//!
//! The console runs any number of groups in x in one dispatch; D3D12 runs at most 65,535.
//! The title's tile lists are dispatches of one group per 8x4 tile, 115,200 at 2560x1440: in
//! a captured frame one class list asked for 67,440, and cut to 65,535 it lost its last 1,905
//! tiles, the lowest of the picture (`victory-gpu`'s `indirect::widest`). With the switch on,
//! the host runs such a dispatch in parts and tells each part's program the number of its
//! first group in a user-data word of kind [`USER_GROUP_BASE`]; the program adds the word to
//! the group number it reads in x. Outside a dispatch in parts the word is 0.
//!
//! Only compute programs that read their group number in x alone, as world's: the title's
//! lists are one-dimensional. Not the programs with a spilled user-data table (the structure
//! builder's, counted in threads: `lower::counted_in_threads`). Not a program whose host root
//! signature has no room for one more word: lowered with it, it is lowered again without it
//! and keeps the host's cut ([`take_no_room`] tells the host, which warns).

use std::cell::Cell;

use super::UserData;
use super::lower::{Lowered, Parts};
use crate::ShaderError;
use crate::io::Input;
use crate::native::{HwStage, RootType, StageIo};

/// `RootType::Other` code of a user-data dword the host sets: the number of the first thread
/// group in x of the part of a dispatch that runs (0 for a dispatch run whole). Asked for by
/// the programs [`takes_base`] picks while [`split_dispatch`] is on. (World's code for it,
/// 0x104, is `USER_ROOT_ADDRESS` here.)
pub const USER_GROUP_BASE: u32 = 0x107;

/// The dwords of a host root signature.
const ROOT_DWORDS: u32 = 64;

thread_local! {
    /// [`split_dispatch`] for the tests of one thread.
    static FOR_TESTS: Cell<Option<bool>> = const { Cell::new(None) };
    /// Whether the program lowered on this thread may take the word: not while it is lowered
    /// again for want of room ([`lower`]).
    static ROOM: Cell<bool> = const { Cell::new(true) };
    /// Whether a program lowered on this thread since [`take_no_room`] went without the word
    /// for want of room.
    static NO_ROOM: Cell<bool> = const { Cell::new(false) };
}

/// `VICTORY_SPLIT_DISPATCH=1`: compute programs that read their group number in x alone add
/// the number of their part's first group ([`USER_GROUP_BASE`]), and the host runs a dispatch
/// of more than 65,535 groups in x in parts. Off by default: no program asks for the word, and
/// the host splits nothing. Read once. Part of the translation: the shader cache keys on it
/// (`victory-gpu`'s does, while it is on).
pub fn split_dispatch() -> bool {
    #[cfg(test)]
    if let Some(on) = FOR_TESTS.with(Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_DISPATCH").is_some_and(|v| v == "1"))
}

/// Sets [`split_dispatch`] for the tests of this thread (`None`: as the environment says).
#[cfg(test)]
pub(super) fn set_for_tests(on: Option<bool>) {
    FOR_TESTS.with(|c| c.set(on));
}

/// Whether the program of `parts` adds [`USER_GROUP_BASE`] to the group number it reads in x.
pub(super) fn takes_base(parts: &Parts) -> bool {
    let reads = |c| parts.io.inputs.iter().any(|i| i.1 == Input::GroupId(c));
    split_dispatch()
        && ROOM.with(Cell::get)
        && parts.native.stage == HwStage::Cs
        && matches!(parts.native.io, StageIo::Cs(_))
        && reads(0)
        && !reads(1)
        && !reads(2)
        && !crate::spill::is_spilled(parts.native)
}

/// The most dwords a host root signature takes for a program of `user_data` and `views` root
/// views: its user data, rounded up to 4 dwords as the host declares it
/// (`HostShader::user_data_dwords`), and 2 for each root view (1 where the host makes it a
/// table of one view).
fn root_dwords(user_data: &[UserData], views: usize) -> u32 {
    let user = user_data.iter().map(|u| u.index + 1).max().unwrap_or(0).next_multiple_of(4);
    user + 2 * views as u32
}

/// `lower::lower_counted`, with the program's first group where it takes one and its root
/// signature has room for it; else lowered again without it.
pub(super) fn lower(parts: &Parts, counted: bool) -> Result<Lowered, ShaderError> {
    let l = super::lower::lower_counted(parts, counted)?;
    let took = l.user_data.iter().any(|u| u.root == RootType::Other(USER_GROUP_BASE));
    let views = l.root_cbvs.len() + l.root_srvs.len() + l.root_uavs.len();
    if !took || root_dwords(&l.user_data, views) <= ROOT_DWORDS {
        return Ok(l);
    }
    NO_ROOM.with(|c| c.set(true));
    ROOM.with(|c| c.set(false));
    let again = super::lower::lower_counted(parts, counted);
    ROOM.with(|c| c.set(true));
    again
}

/// Whether a program translated on this thread since the last call went without its first
/// group for want of room in its root signature: its dispatches of more than 65,535 groups in
/// x are cut, as without the switch. Cleared by the call.
pub fn take_no_room() -> bool {
    NO_ROOM.with(|c| c.replace(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The room `user` dwords of user data and `views` root views take.
    fn room(user: u32, views: usize) -> u32 {
        let user: Vec<UserData> = (0..user).map(|index| UserData { index, slot: 0, root: RootType::Constants, dword: index }).collect();
        root_dwords(&user, views)
    }

    /// The room is counted as the host declares the root signature: user data in fours, two
    /// dwords a root view.
    #[test]
    fn the_root_signature_is_counted_as_the_host_makes_it() {
        assert_eq!(room(0, 0), 0);
        assert_eq!(room(5, 0), 8);
        assert_eq!(room(8, 3), 14);
        // 61 dwords of user data are declared as 64, which fit; 57 (60) and four views do not.
        assert!(room(61, 0) <= ROOT_DWORDS);
        assert!(room(57, 4) > ROOT_DWORDS);
    }

    /// Off unless the switch is 1; the tests' own setting stays in their thread.
    #[test]
    fn the_switch_is_set_by_its_tests_alone() {
        set_for_tests(Some(true));
        assert!(split_dispatch());
        set_for_tests(Some(false));
        assert!(!split_dispatch());
        set_for_tests(None);
        assert!(!take_no_room());
    }
}
