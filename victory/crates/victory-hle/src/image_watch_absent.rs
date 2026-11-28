//! A watch over the title's main image once it is mapped: absent from this
//! tree.
//!
//! The private build has a module here, started when the title's main image
//! is recorded ([`crate::runtime::set_main_image`]), that is made from what
//! the title's code is. The public tree leaves it out. What is here only
//! gives the caller what it needs, and does nothing.

/// The title's main image is mapped at `_base`: nothing is started.
pub(crate) fn start(_base: usize) {}
