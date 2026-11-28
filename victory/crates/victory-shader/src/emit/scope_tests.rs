//! A translation that panics inside one of the lowering's scoped forms (pair meetings, local
//! waves, every lane per host wave, first-half pairs) leaves that form off for the next program
//! on its thread: a pipeline build catches the panic and its worker goes on
//! (`victory-gpu`'s `pipeline::make`), and the next program translated there was lowered in the
//! form and cached under the key of the plain one.

use super::super::caught;
use super::*;

#[test]
fn a_panic_in_a_translation_leaves_the_lowerings_forms_as_they_were() {
    caught(|| with_pair_meetings(|| panic!("a translation's panic")));
    assert!(!PAIR_MEETINGS.get(), "pair meetings");
    caught(|| with_local_waves(|| panic!("a translation's panic")));
    assert!(!LOCAL_WAVES.get(), "local waves");
    caught(|| with_local_waves_any(|| panic!("a translation's panic")));
    assert!(!LOCAL_ANY.get() && !LOCAL_WAVES.get(), "every lane per host wave");
    caught(|| with_first_half_pairs(|| panic!("a translation's panic")));
    assert!(!FIRST_HALF_PAIRS.get(), "first-half pairs");
}
