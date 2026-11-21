use crate::diagnostic::internal::DiagnosticsRecorder;
use crate::render_phase::TrackedRenderPass;
use crate::render_resource::{CommandEncoder, RenderPassDescriptor};
use crate::renderer::{wgpu_wrapper, RenderDevice};
use alloc::borrow::Cow;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::change_detection::Tick;
use bevy_ecs::component::ComponentId;
use bevy_ecs::prelude::*;
use bevy_ecs::query::{QueryData, QueryFilter, QueryState};
use bevy_ecs::system::{
    Deferred, SystemAccess, SystemBuffer, SystemMeta, SystemName, SystemParam,
    SystemParamValidationError,
};
use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
use bevy_ecs::world::DeferredWorld;
#[cfg(feature = "trace")]
use bevy_log::info_span;
#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::ComputeTaskPool;
use core::marker::PhantomData;
use wgpu::CommandBuffer;

#[derive(Default)]
struct PendingCommandBuffersInner {
    commands: Vec<PendingCommandBuffer>,
}

enum PendingCommandBuffer {
    Buffer(CommandBuffer),
    Encoder {
        encoder: CommandEncoder,
        #[cfg(feature = "trace")]
        name: Cow<'static, str>,
    },
}

wgpu_wrapper!(struct WgpuPendingCommandBuffersInner(PendingCommandBuffersInner));

/// A resource that holds command buffers and encoders that are pending submission to the render queue.
#[derive(Resource)]
pub struct PendingCommandBuffers(WgpuPendingCommandBuffersInner);

impl Default for PendingCommandBuffers {
    fn default() -> Self {
        Self(WgpuPendingCommandBuffersInner::new(
            PendingCommandBuffersInner::default(),
        ))
    }
}

impl PendingCommandBuffers {
    pub fn push(&mut self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        self.0
            .commands
            .extend(buffers.into_iter().map(PendingCommandBuffer::Buffer));
    }

    fn append(&mut self, commands: &mut Vec<PendingCommandBuffer>) {
        self.0.commands.append(commands);
    }

    pub fn push_encoder(&mut self, encoder: CommandEncoder, name: impl Into<Cow<'static, str>>) {
        #[cfg(not(feature = "trace"))]
        let _ = name;

        self.0.commands.push(PendingCommandBuffer::Encoder {
            encoder,
            #[cfg(feature = "trace")]
            name: name.into(),
        });
    }

    pub fn finish(&mut self) -> impl Iterator<Item = CommandBuffer> {
        #[cfg(feature = "trace")]
        let _finish_command_buffers_span = info_span!("finish_command_buffers").entered();

        let commands = self.0.commands.drain(..);

        #[cfg(target_arch = "wasm32")]
        {
            finish_sequential(commands)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            finish_parallel(commands)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.commands.len()
    }
}

/// Finishes pending command buffers sequentially, preserving their order.
///
/// Used on wasm, where wgpu command encoders and buffers are `!Send` and so
/// cannot be finished across task pool threads.
#[cfg(target_arch = "wasm32")]
fn finish_sequential(
    commands: impl Iterator<Item = PendingCommandBuffer>,
) -> impl Iterator<Item = CommandBuffer> {
    commands.into_iter().map(|command| match command {
        PendingCommandBuffer::Buffer(command_buffer) => command_buffer,
        PendingCommandBuffer::Encoder { encoder, .. } => encoder.finish(),
    })
}

/// Finishes pending command encoders in parallel on the [`ComputeTaskPool`],
/// then reassembles the command buffers in their original order.
///
/// Every render system (and view) records into its own encoder, so a frame
/// would otherwise submit one command buffer per system and view; on Metal each
/// is at least one `MTLCommandBuffer` with its own creation, commit and
/// completion cost. Before finishing, each run of consecutive encoders is merged
/// into at most one encoder per compute thread with [`CommandEncoder::append`],
/// so finishing still runs in parallel. Submission order is unchanged.
#[cfg(not(target_arch = "wasm32"))]
fn finish_parallel(
    commands: impl Iterator<Item = PendingCommandBuffer>,
) -> impl Iterator<Item = CommandBuffer> {
    let chunks = ComputeTaskPool::get().thread_num().max(1);
    let commands = merge_encoder_runs(commands.collect(), chunks, |dst, src| {
        let (
            PendingCommandBuffer::Encoder { encoder: dst, .. },
            PendingCommandBuffer::Encoder {
                encoder: src,
                #[cfg(feature = "trace")]
                name,
            },
        ) = (dst, src)
        else {
            unreachable!("only encoders are merged")
        };
        dst.append(src)
            .map_err(|encoder| PendingCommandBuffer::Encoder {
                encoder,
                #[cfg(feature = "trace")]
                name,
            })
    });
    let mut command_buffers = Vec::with_capacity(commands.len());
    let mut finished_encoders = ComputeTaskPool::get().scope(|scope| {
        for (index, command) in commands.into_iter().enumerate() {
            match command {
                PendingCommandBuffer::Buffer(command_buffer) => {
                    command_buffers.push((index, command_buffer));
                }
                PendingCommandBuffer::Encoder {
                    encoder,
                    #[cfg(feature = "trace")]
                    name,
                } => {
                    scope.spawn(async move {
                        #[cfg(feature = "trace")]
                        let _span =
                            info_span!("finish_command_buffer", system = name.as_ref()).entered();
                        (index, encoder.finish())
                    });
                }
            }
        }
    });

    command_buffers.append(&mut finished_encoders);
    command_buffers.sort_unstable_by_key(|(index, _)| *index);
    command_buffers
        .into_iter()
        .map(|(_, command_buffer)| command_buffer)
}

/// Something [`merge_encoder_runs`] can merge.
#[cfg(any(test, not(target_arch = "wasm32")))]
trait Mergeable {
    /// Whether this item can absorb, or be absorbed by, a neighbour.
    fn is_mergeable(&self) -> bool;
}

#[cfg(not(target_arch = "wasm32"))]
impl Mergeable for PendingCommandBuffer {
    fn is_mergeable(&self) -> bool {
        matches!(self, PendingCommandBuffer::Encoder { .. })
    }
}

/// Merges each run of consecutive mergeable items into at most `chunks` items,
/// about the same number of source items each, keeping their order. `append`
/// moves `src` to the end of `dst`, or hands `src` back if it cannot; the item
/// it hands back then starts the next merged item.
#[cfg(any(test, not(target_arch = "wasm32")))]
fn merge_encoder_runs<T: Mergeable>(
    items: Vec<T>,
    chunks: usize,
    mut append: impl FnMut(&mut T, T) -> Result<(), T>,
) -> Vec<T> {
    let mut out = Vec::with_capacity(items.len());
    let mut run: Vec<T> = Vec::new();
    let mut flush = |run: &mut Vec<T>, out: &mut Vec<T>| {
        let per_chunk = run.len().div_ceil(chunks.max(1));
        let mut head: Option<(T, usize)> = None;
        for item in run.drain(..) {
            head = match head.take() {
                Some((mut dst, n)) if n < per_chunk => match append(&mut dst, item) {
                    Ok(()) => Some((dst, n + 1)),
                    Err(item) => {
                        out.push(dst);
                        Some((item, 1))
                    }
                },
                previous => {
                    out.extend(previous.map(|(dst, _)| dst));
                    Some((item, 1))
                }
            };
        }
        out.extend(head.map(|(dst, _)| dst));
    };
    for item in items {
        if item.is_mergeable() {
            run.push(item);
        } else {
            flush(&mut run, &mut out);
            out.push(item);
        }
    }
    flush(&mut run, &mut out);
    out
}

#[derive(Default)]
struct RenderContextStateInner {
    command_encoder: Option<CommandEncoder>,
    commands: Vec<PendingCommandBuffer>,
    render_device: Option<RenderDevice>,
}

impl RenderContextStateInner {
    fn flush_encoder(&mut self) {
        if let Some(encoder) = self.command_encoder.take() {
            self.commands.push(PendingCommandBuffer::Encoder {
                encoder,
                #[cfg(feature = "trace")]
                name: "RenderContextState::flush_encoder".into(),
            });
        }
    }
}

wgpu_wrapper!(struct WgpuRenderContextStateInner(RenderContextStateInner));

/// A resource that holds the current render context state, including command encoder and command buffers.
/// This is used internally by the [`RenderContext`] system parameter. Implements [`SystemBuffer`] to
/// append command buffers and unfinished encoders in topological system order. Pending encoders are
/// finished in parallel immediately before submission.
pub struct RenderContextState(WgpuRenderContextStateInner);

impl Default for RenderContextState {
    fn default() -> Self {
        Self(WgpuRenderContextStateInner::new(
            RenderContextStateInner::default(),
        ))
    }
}

impl RenderContextState {
    fn flush_encoder(&mut self) {
        self.0.flush_encoder();
    }

    fn command_encoder(&mut self, label: &str) -> &mut CommandEncoder {
        let render_device = self
            .0
            .render_device
            .clone()
            .expect("RenderDevice must be set before accessing command_encoder");

        self.0.command_encoder.get_or_insert_with(|| {
            render_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        })
    }
}

impl SystemBuffer for RenderContextState {
    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        #[cfg(feature = "trace")]
        let _span =
            info_span!("RenderContextState::apply", system = %_system_meta.name()).entered();

        let inner = &mut *self.0;

        inner.flush_encoder();

        #[cfg(feature = "trace")]
        for command in &mut inner.commands {
            if let PendingCommandBuffer::Encoder { name, .. } = command {
                *name = _system_meta.name().clone().into();
            }
        }

        if !inner.commands.is_empty() {
            let mut pending = world.resource_mut::<PendingCommandBuffers>();
            pending.append(&mut inner.commands);
        }

        inner.render_device = None;
    }
}

/// A system parameter that provides access to a command encoder and render device for issuing
/// rendering commands inside any system running beneath the root [`super::RenderGraph`] schedule in the
/// [`super::render_system`] system.
#[derive(SystemParam)]
pub struct RenderContext<'w, 's> {
    state: Deferred<'s, RenderContextState>,
    system_name: SystemName,
    render_device: Res<'w, RenderDevice>,
    diagnostics_recorder: Option<Res<'w, DiagnosticsRecorder>>,
}

impl<'w, 's> RenderContext<'w, 's> {
    fn ensure_device(&mut self) {
        if self.state.0.render_device.is_none() {
            self.state.0.render_device = Some(self.render_device.clone());
        }
    }

    /// Returns the render device associated with this render context.
    pub fn render_device(&self) -> &RenderDevice {
        &self.render_device
    }

    /// Returns the diagnostics recorder, if available.
    pub fn diagnostic_recorder(&self) -> Option<Res<'w, DiagnosticsRecorder>> {
        self.diagnostics_recorder.as_ref().map(Res::clone)
    }

    /// Returns the current command encoder, creating one if it does not already exist.
    pub fn command_encoder(&mut self) -> &mut CommandEncoder {
        self.ensure_device();
        self.state.command_encoder(self.system_name.as_str())
    }

    /// Begins a tracked render pass with the given descriptor.
    pub fn begin_tracked_render_pass<'a>(
        &'a mut self,
        descriptor: RenderPassDescriptor<'_>,
    ) -> TrackedRenderPass<'a> {
        self.ensure_device();

        let command_encoder = self.state.0.command_encoder.get_or_insert_with(|| {
            self.render_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(self.system_name.as_str()),
                })
        });

        let render_pass = command_encoder.begin_render_pass(&descriptor);
        TrackedRenderPass::new(&self.render_device, render_pass)
    }

    /// Adds a finished command buffer to be submitted later.
    pub fn add_command_buffer(&mut self, command_buffer: CommandBuffer) {
        self.state.flush_encoder();
        self.state
            .0
            .commands
            .push(PendingCommandBuffer::Buffer(command_buffer));
    }
}

/// A system parameter that can be used to explicitly flush pending command buffers to the render queue.
/// This is typically not necessary, as command buffers are automatically flushed at the end of each
/// render system. However, in some cases it may be useful to flush command buffers earlier.
#[derive(SystemParam)]
pub struct FlushCommands<'w> {
    pending: ResMut<'w, PendingCommandBuffers>,
    queue: Res<'w, super::RenderQueue>,
}

impl<'w> FlushCommands<'w> {
    /// Flushes all pending command buffers to the render queue.
    pub fn flush(&mut self) {
        let mut buffers = self.pending.finish().peekable();
        if buffers.peek().is_some() {
            self.queue.submit(buffers);
        }
    }
}

/// The entity corresponding to the current view being rendered.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Deref, DerefMut)]
pub struct CurrentView(pub Entity);

/// A query that fetches components for the entity corresponding to the current view being rendered,
/// as defined by the [`CurrentView`] resource, equivalent to `query.get(current_view.entity())`.
pub struct ViewQuery<'w, 's, D: QueryData, F: QueryFilter = ()> {
    entity: Entity,
    item: D::Item<'w, 's>,
    _filter: PhantomData<F>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> ViewQuery<'w, 's, D, F> {
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    pub fn into_inner(self) -> D::Item<'w, 's> {
        self.item
    }
}

pub struct ViewQueryState<D: QueryData, F: QueryFilter> {
    resource_id: ComponentId,
    query_state: QueryState<D, F>,
}

// SAFETY: ViewQuery accesses the CurrentView resource (read) and query components.
// Access is properly registered in init_access.
unsafe impl<'a, D: QueryData + 'static, F: QueryFilter + 'static> SystemParam
    for ViewQuery<'a, '_, D, F>
{
    type State = ViewQueryState<D, F>;
    type Item<'w, 's> = ViewQuery<'w, 's, D, F>;

    fn init_state(world: &mut World) -> Self::State {
        ViewQueryState {
            resource_id: world
                .components_registrator()
                .register_component::<CurrentView>(),
            query_state: QueryState::new(world),
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        system_access: &mut SystemAccess,
        world: &mut World,
    ) {
        let component_access_set = system_access.require_shared_access::<Self>(system_meta);
        component_access_set.add_resource_read(state.resource_id);

        <Query<'_, '_, D, F> as SystemParam>::init_access(
            &state.query_state,
            system_meta,
            system_access,
            world,
        );
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _system_meta: &SystemMeta,
        world: UnsafeWorldCell<'w>,
        _change_tick: Tick,
    ) -> Result<Self::Item<'w, 's>, SystemParamValidationError> {
        // SAFETY: We have registered resource read access in init_access
        let current_view = unsafe { world.get_resource::<CurrentView>() };

        let Some(current_view) = current_view else {
            return Err(SystemParamValidationError::skipped::<Self>(
                "CurrentView resource not present",
            ));
        };

        let entity = current_view.entity();

        // SAFETY: Query state access is properly registered in init_access.
        // The caller ensures the world matches the one used in init_state.
        let item = unsafe { state.query_state.get_unchecked(world, entity) }.map_err(|_| {
            SystemParamValidationError::skipped::<Self>("Current view entity does not match query")
        })?;

        Ok(ViewQuery {
            entity,
            item,
            _filter: PhantomData,
        })
    }
}

// SAFETY: ViewQuery with ReadOnlyQueryData only reads from the world.
unsafe impl<'w, 's, D: bevy_ecs::query::ReadOnlyQueryData + 'static, F: QueryFilter + 'static>
    bevy_ecs::system::ReadOnlySystemParam for ViewQuery<'w, 's, D, F>
{
}

#[cfg(test)]
mod tests {
    use super::{merge_encoder_runs, Mergeable};
    use alloc::{vec, vec::Vec};

    /// An encoder stand-in: the ids of the source encoders it holds, in order.
    /// `Fixed` items are finished command buffers, which never merge.
    #[derive(Debug, PartialEq)]
    enum Item {
        Encoder(Vec<u32>),
        Fixed(u32),
    }

    impl Mergeable for Item {
        fn is_mergeable(&self) -> bool {
            matches!(self, Item::Encoder(_))
        }
    }

    fn encoders(ids: core::ops::Range<u32>) -> Vec<Item> {
        ids.map(|id| Item::Encoder(vec![id])).collect()
    }

    fn append(dst: &mut Item, src: Item) -> Result<(), Item> {
        match (dst, src) {
            (Item::Encoder(dst), Item::Encoder(src)) => {
                dst.extend(src);
                Ok(())
            }
            (_, src) => Err(src),
        }
    }

    #[test]
    fn a_run_is_split_into_chunks_in_order() {
        let merged = merge_encoder_runs(encoders(0..10), 4, append);
        assert_eq!(
            merged,
            vec![
                Item::Encoder(vec![0, 1, 2]),
                Item::Encoder(vec![3, 4, 5]),
                Item::Encoder(vec![6, 7, 8]),
                Item::Encoder(vec![9]),
            ]
        );
    }

    #[test]
    fn finished_buffers_split_runs_and_keep_their_place() {
        let mut items = encoders(0..3);
        items.push(Item::Fixed(100));
        items.extend(encoders(3..5));
        let merged = merge_encoder_runs(items, 1, append);
        assert_eq!(
            merged,
            vec![
                Item::Encoder(vec![0, 1, 2]),
                Item::Fixed(100),
                Item::Encoder(vec![3, 4]),
            ]
        );
    }

    #[test]
    fn a_refused_append_starts_the_next_chunk() {
        // Encoder 2 cannot be absorbed (for example it is invalid).
        let refuse_2 = |dst: &mut Item, src: Item| match &src {
            Item::Encoder(ids) if ids == &vec![2] => Err(src),
            _ => append(dst, src),
        };
        let merged = merge_encoder_runs(encoders(0..5), 1, refuse_2);
        assert_eq!(
            merged,
            vec![Item::Encoder(vec![0, 1]), Item::Encoder(vec![2, 3, 4])]
        );
    }

    #[test]
    fn single_items_and_empty_input_pass_through() {
        assert_eq!(merge_encoder_runs(Vec::<Item>::new(), 4, append), vec![]);
        assert_eq!(
            merge_encoder_runs(vec![Item::Fixed(7)], 4, append),
            vec![Item::Fixed(7)]
        );
        assert_eq!(
            merge_encoder_runs(encoders(0..1), 4, append),
            vec![Item::Encoder(vec![0])]
        );
    }

    #[test]
    fn more_chunks_than_encoders_keeps_one_per_encoder() {
        assert_eq!(
            merge_encoder_runs(encoders(0..3), 8, append),
            encoders(0..3)
        );
    }
}
