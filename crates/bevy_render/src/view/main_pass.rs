//! Opt-in control over how the 3D main pass uses its multisampled attachments.
//!
//! By default the core 3D pipeline draws the opaque, alpha-mask and transparent phases in two
//! render passes over the same multisampled (MSAA) color and depth attachments. Each pass loads
//! and stores the full multisampled textures and resolves them, although nothing reads the
//! individual samples once the main pass is done. On tile-based GPUs (Apple silicon, most mobile
//! GPUs) every one of those loads and stores is a round trip of the whole multisampled image
//! through memory.
//!
//! [`MergedMainPass`] asks the core 3D pipeline to draw all of those phases in a single render
//! pass that resolves at the end and discards the multisampled samples instead of storing them.
//! [`TransientMsaa`] additionally creates the multisampled attachments as
//! [`TextureUsages::TRANSIENT_ATTACHMENT`], which lets the driver keep them in tile memory only
//! (memoryless textures on Metal).
//!
//! Both are hints: every frame the renderer decides per view whether the merged layout is safe and
//! falls back to the regular split layout when it is not (see [`MainPassLayout`]).

use super::{main_texture_view_formats, Msaa, ViewTarget};
use crate::{
    camera::ExtractedCamera,
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    RenderApp,
};
use bevy_ecs::prelude::*;
use bevy_extract_macros::ExtractComponent;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use wgpu::{TextureDescriptor, TextureDimension, TextureUsages};

/// Opts a 3D camera into drawing its whole main pass in one render pass when MSAA is enabled.
///
/// With this component, the opaque, alpha-mask, skybox and transparent phases of a multisampled
/// view are drawn in a single render pass. The multisampled color attachment is resolved at the
/// end of that pass and its samples are discarded ([`StoreOp::Discard`](wgpu::StoreOp::Discard))
/// instead of being written back to memory. The multisampled depth attachment is discarded too
/// when nothing can read it later: it has neither [`TextureUsages::TEXTURE_BINDING`] nor
/// [`TextureUsages::COPY_SRC`], and no other camera renders into the same depth texture.
///
/// The final image is the same as with the default split layout. The saving is memory bandwidth,
/// which matters most on tile-based GPUs, where the default layout writes the multisampled
/// attachments out after the opaque pass, reads them back for the transparent pass and writes
/// them out again at the end.
///
/// This is a request, not a guarantee. The main pass stays split for a frame when:
/// - MSAA is off for the view ([`Msaa::Off`]); there is nothing to merge,
/// - order-independent transparency is enabled,
/// - the view has items in a phase drawn between the opaque and transparent passes or after
///   them into the multisampled attachments (screen-space specular transmission, wireframes,
///   the atmosphere sky pass),
/// - a later camera on the same render target loads the multisampled color without
///   [`MsaaWriteback`](bevy_camera::MsaaWriteback),
/// - the camera, or a render-world system, requests the split layout with
///   [`MainPassSplitRequired`] or [`MainPassLayout::require_split`].
///
/// Only the core 3D pipeline honors this component; it has no effect on 2D cameras.
///
/// See [`TransientMsaa`] to also keep the multisampled attachments out of memory entirely.
#[derive(Component, Default, Clone, Copy, ExtractComponent, Reflect, PartialEq, Eq, Debug)]
#[reflect(Component, Default, PartialEq, Debug)]
#[extract_app(RenderApp)]
pub struct MergedMainPass;

/// Creates the multisampled main-pass attachments of a camera as transient attachments.
///
/// Transient attachments ([`TextureUsages::TRANSIENT_ATTACHMENT`]) can only be cleared at the
/// start of a render pass and discarded at its end, never loaded or stored. Tile-based GPUs can
/// then keep them entirely in tile memory: on Metal they are memoryless textures, so a 4x MSAA
/// [`Hdr`](bevy_camera::Hdr) color and depth attachment at 2560x1440 no longer take about
/// 177 MB of GPU memory. Backends that do not benefit treat the flag as a no-op (see
/// `AdapterInfo::transient_saves_memory`).
///
/// This requires [`MergedMainPass`], since only a single main pass can work with attachments
/// that are never stored. The color attachment is transient when the view is the only one
/// rendering into its main texture, clears it (its [`ClearColorConfig`] is not
/// [`ClearColorConfig::None`]) and does not use [`MsaaWriteback::Always`]. The depth attachment is
/// transient when its usage is exactly [`TextureUsages::RENDER_ATTACHMENT`] (no depth prepass,
/// no occlusion culling, nothing else sampling or copying it), the camera clears depth and no
/// other camera uses the same depth texture.
///
/// If the main pass of a frame has to be split after all (see [`MergedMainPass`]), the view
/// gets a regular multisampled color texture for that frame. Rendering stays correct, but the
/// memory saving is lost while the split layout is needed.
///
/// [`ClearColorConfig`]: bevy_camera::ClearColorConfig
/// [`ClearColorConfig::None`]: bevy_camera::ClearColorConfig::None
/// [`MsaaWriteback::Always`]: bevy_camera::MsaaWriteback::Always
#[derive(Component, Default, Clone, Copy, ExtractComponent, Reflect, PartialEq, Eq, Debug)]
#[reflect(Component, Default, PartialEq, Debug)]
#[extract_app(RenderApp)]
#[require(MergedMainPass)]
pub struct TransientMsaa;

/// Keeps the main pass of a camera split into separate opaque and transparent render passes,
/// even if it has [`MergedMainPass`].
///
/// Insert this on a camera when one of your render systems needs the multisampled main-pass
/// attachments between the opaque and transparent passes, or loads them after the main pass:
/// for example a system ordered after `main_opaque_pass_3d` and before
/// `main_transparent_pass_3d`, or one that renders into [`ViewTarget::get_color_attachment`]
/// after the main pass. Render-world systems that only need the split layout for some frames
/// can instead call [`MainPassLayout::require_split`] in [`MainPassLayoutSystems::Restrict`].
#[derive(Component, Default, Clone, Copy, ExtractComponent, Reflect, PartialEq, Eq, Debug)]
#[reflect(Component, Default, PartialEq, Debug)]
#[extract_app(RenderApp)]
pub struct MainPassSplitRequired;

/// How the main pass of a 3D view is laid out this frame.
///
/// This render-world component is recomputed every frame, in
/// [`MainPassLayoutSystems::Decide`], for every 3D view. Views without it use the split layout.
#[derive(Component, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum MainPassLayout {
    /// The opaque and transparent phases are drawn in separate render passes, and the
    /// multisampled attachments are stored after each of them. This is the default.
    #[default]
    Split,
    /// The opaque, alpha-mask, skybox and transparent phases are drawn in one render pass.
    /// The multisampled color attachment is resolved and discarded at the end of that pass.
    Merged {
        /// Another view renders into the same depth texture after this one, so the depth
        /// attachment must be stored.
        keep_depth: bool,
    },
}

impl MainPassLayout {
    /// Returns `true` if the main pass is drawn as one render pass.
    #[inline]
    pub fn is_merged(&self) -> bool {
        matches!(self, Self::Merged { .. })
    }

    /// Forces the split layout for this frame.
    ///
    /// Call this from a system in [`MainPassLayoutSystems::Restrict`] when the view needs the
    /// multisampled attachments of the main pass between the opaque and transparent passes or
    /// after the main pass, for example because a phase drawn there has items this frame.
    #[inline]
    pub fn require_split(&mut self) {
        *self = Self::Split;
    }
}

/// System sets that decide the [`MainPassLayout`] of each view, in order.
///
/// They run in [`RenderSystems::Prepare`](crate::RenderSystems::Prepare), after the render
/// phases are queued and sorted and before
/// [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MainPassLayoutSystems {
    /// The core pipeline writes each view's [`MainPassLayout`] from the camera settings.
    Decide,
    /// Plugins call [`MainPassLayout::require_split`] for views that need the split layout
    /// this frame.
    Restrict,
    /// The multisampled attachments are made consistent with the final layout.
    Finalize,
}

/// Returns `true` if a view may get a transient multisampled color attachment from
/// [`prepare_view_targets`](super::prepare_view_targets).
///
/// `views_sharing_texture` is the number of views rendering into the same main texture.
pub(crate) fn transient_msaa_color_allowed(
    camera: &ExtractedCamera,
    msaa: Msaa,
    views_sharing_texture: usize,
) -> bool {
    msaa.samples() > 1
        && views_sharing_texture == 1
        && !matches!(camera.clear_color, bevy_camera::ClearColorConfig::None)
        && camera.msaa_writeback != bevy_camera::MsaaWriteback::Always
}

/// Replaces the transient multisampled color attachment of views whose main pass is split this
/// frame with a regular one, since a split main pass has to store the attachment in between.
pub fn finalize_transient_msaa_textures(
    mut views: Query<(&mut ViewTarget, Option<&MainPassLayout>)>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
) {
    for (mut view_target, layout) in &mut views {
        if layout.is_some_and(MainPassLayout::is_merged) {
            continue;
        }
        let Some(sampled) = view_target.sampled_main_texture() else {
            continue;
        };
        if !sampled
            .usage()
            .contains(TextureUsages::TRANSIENT_ATTACHMENT)
        {
            continue;
        }
        let replacement: CachedTexture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("main_texture_sampled"),
                size: sampled.size(),
                mip_level_count: 1,
                sample_count: sampled.sample_count(),
                dimension: TextureDimension::D2,
                format: sampled.format(),
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: main_texture_view_formats(sampled.format()),
            },
        );
        view_target.replace_sampled_main_texture(replacement);
    }
}
