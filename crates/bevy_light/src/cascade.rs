//! Provides shadow cascade configuration and construction helpers.

use bevy_camera::{Camera, Projection};
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_math::{ops, Mat4, Vec3A, Vec4};
use bevy_reflect::prelude::*;
use bevy_transform::components::GlobalTransform;

use crate::{DirectionalLight, DirectionalLightShadowMap};

/// Controls how cascaded shadow mapping works.
/// Prefer using [`CascadeShadowConfigBuilder`] to construct an instance.
///
/// ```
/// # use bevy_light::CascadeShadowConfig;
/// # use bevy_light::CascadeShadowConfigBuilder;
/// # use bevy_utils::default;
/// #
/// let config: CascadeShadowConfig = CascadeShadowConfigBuilder {
///   maximum_distance: 100.0,
///   ..default()
/// }.into();
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct CascadeShadowConfig {
    /// The (positive) distance to the far boundary of each cascade.
    pub bounds: Vec<f32>,
    /// The proportion of overlap each cascade has with the previous cascade.
    pub overlap_proportion: f32,
    /// The (positive) distance to the near boundary of the first cascade.
    pub minimum_distance: f32,
}

impl Default for CascadeShadowConfig {
    fn default() -> Self {
        CascadeShadowConfigBuilder::default().into()
    }
}

fn calculate_cascade_bounds(
    num_cascades: usize,
    nearest_bound: f32,
    shadow_maximum_distance: f32,
) -> Vec<f32> {
    if num_cascades == 1 {
        return vec![shadow_maximum_distance];
    }
    let base = ops::powf(
        shadow_maximum_distance / nearest_bound,
        1.0 / (num_cascades - 1) as f32,
    );
    (0..num_cascades)
        .map(|i| nearest_bound * ops::powf(base, i as f32))
        .collect()
}

/// Builder for [`CascadeShadowConfig`].
#[derive(Clone)]
pub struct CascadeShadowConfigBuilder {
    /// The number of shadow cascades.
    /// More cascades increases shadow quality by mitigating perspective aliasing - a phenomenon where areas
    /// nearer the camera are covered by fewer shadow map texels than areas further from the camera, causing
    /// blocky looking shadows.
    ///
    /// This does come at the cost increased rendering overhead, however this overhead is still less
    /// than if you were to use fewer cascades and much larger shadow map textures to achieve the
    /// same quality level.
    ///
    /// In case rendered geometry covers a relatively narrow and static depth relative to camera, it may
    /// make more sense to use fewer cascades and a higher resolution shadow map texture as perspective aliasing
    /// is not as much an issue. Be sure to adjust `minimum_distance` and `maximum_distance` appropriately.
    pub num_cascades: usize,
    /// The minimum shadow distance, which can help improve the texel resolution of the first cascade.
    /// Areas nearer to the camera than this will likely receive no shadows.
    ///
    /// NOTE: Due to implementation details, this usually does not impact shadow quality as much as
    /// `first_cascade_far_bound` and `maximum_distance`. At many view frustum field-of-views, the
    /// texel resolution of the first cascade is dominated by the width / height of the view frustum plane
    /// at `first_cascade_far_bound` rather than the depth of the frustum from `minimum_distance` to
    /// `first_cascade_far_bound`.
    pub minimum_distance: f32,
    /// The maximum shadow distance.
    /// Areas further from the camera than this will likely receive no shadows.
    pub maximum_distance: f32,
    /// Sets the far bound of the first cascade, relative to the view origin.
    /// In-between cascades will be exponentially spaced relative to the maximum shadow distance.
    /// NOTE: This is ignored if there is only one cascade, the maximum distance takes precedence.
    pub first_cascade_far_bound: f32,
    /// Sets the overlap proportion between cascades.
    /// The overlap is used to make the transition from one cascade's shadow map to the next
    /// less abrupt by blending between both shadow maps.
    pub overlap_proportion: f32,
}

impl CascadeShadowConfigBuilder {
    /// Returns the cascade config as specified by this builder.
    pub fn build(&self) -> CascadeShadowConfig {
        assert!(
            self.num_cascades > 0,
            "num_cascades must be positive, but was {}",
            self.num_cascades
        );
        assert!(
            self.minimum_distance >= 0.0,
            "maximum_distance must be non-negative, but was {}",
            self.minimum_distance
        );
        assert!(
            self.num_cascades == 1 || self.minimum_distance < self.first_cascade_far_bound,
            "minimum_distance must be less than first_cascade_far_bound, but was {}",
            self.minimum_distance
        );
        assert!(
            self.maximum_distance > self.minimum_distance,
            "maximum_distance must be greater than minimum_distance, but was {}",
            self.maximum_distance
        );
        assert!(
            (0.0..1.0).contains(&self.overlap_proportion),
            "overlap_proportion must be in [0.0, 1.0) but was {}",
            self.overlap_proportion
        );
        CascadeShadowConfig {
            bounds: calculate_cascade_bounds(
                self.num_cascades,
                self.first_cascade_far_bound,
                self.maximum_distance,
            ),
            overlap_proportion: self.overlap_proportion,
            minimum_distance: self.minimum_distance,
        }
    }
}

impl Default for CascadeShadowConfigBuilder {
    fn default() -> Self {
        // The defaults are chosen to be similar to be Unity, Unreal, and Godot.
        // Unity: first cascade far bound = 10.05, maximum distance = 150.0
        // Unreal Engine 5: maximum distance = 200.0
        // Godot: first cascade far bound = 10.0, maximum distance = 100.0
        Self {
            // Currently only support one cascade in WebGL 2.
            num_cascades: if cfg!(all(
                feature = "webgl",
                target_arch = "wasm32",
                not(feature = "webgpu")
            )) {
                1
            } else {
                4
            },
            minimum_distance: 0.1,
            maximum_distance: 150.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
    }
}

impl From<CascadeShadowConfigBuilder> for CascadeShadowConfig {
    fn from(builder: CascadeShadowConfigBuilder) -> Self {
        builder.build()
    }
}

/// A [`DirectionalLight`]'s per-view list of [`Cascade`]s.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct Cascades {
    /// Map from a view to the configuration of each of its [`Cascade`]s.
    pub cascades: EntityHashMap<Vec<Cascade>>,
}

/// A single cascade of a view's shadow map cascade. Several of these are
/// used to cover most of the view to ensure most geometry gets shadows, with
/// some overlap for blending at cascade transitions. Farther away cascades
/// are larger and have a lower effective shadowmap texel per world unit
/// resolution. All cascades have the same pixel dimensions however.
#[derive(Clone, Debug, Default, Reflect)]
#[reflect(Clone, Default)]
pub struct Cascade {
    /// The transform of the light, i.e. the view to world matrix.
    pub world_from_cascade: Mat4,
    /// The orthographic projection for this cascade.
    pub clip_from_cascade: Mat4,
    /// The view-projection matrix for this cascade, converting world space into light clip space.
    /// Importantly, this is derived and stored separately from `view_transform` and `projection` to
    /// ensure shadow stability.
    pub clip_from_world: Mat4,
    /// Size of each shadow map texel in world units.
    pub texel_size: f32,
    /// Kept from an earlier frame by [`CascadeReuse`]: its shadow map can be reused as is.
    pub reused: bool,
}

/// Opt-in for a [`DirectionalLight`]: far cascades take turns redrawing their shadow map instead
/// of all redrawing every frame. A cascade that sits out keeps the matrix it was drawn with, and
/// only while the current view slice still fits inside it.
///
/// Casters in those cascades may be a few frames stale (at most one frame with two such cascades).
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct CascadeReuse {
    /// The first cascade that takes turns. Nearer cascades redraw every frame.
    pub first_cascade: usize,
    /// Widens the cascades that take turns by this fraction, so a moving view stays inside them.
    pub padding: f32,
}

impl Default for CascadeReuse {
    fn default() -> Self {
        Self {
            first_cascade: 2,
            padding: 0.0,
        }
    }
}

impl CascadeReuse {
    /// Depth range added on each side of a turn-taking cascade, as a fraction of its range. Depth
    /// is 32 bit, so this costs no visible precision.
    const DEPTH_PADDING: f32 = 0.25;

    fn takes_turns(&self, cascade: usize) -> bool {
        cascade >= self.first_cascade
    }

    /// Whether it is `cascade`'s turn to redraw in `frame`.
    fn is_due(&self, cascade: usize, cascade_count: usize, frame: u32) -> bool {
        let sharing = cascade_count.saturating_sub(self.first_cascade).max(1);
        !self.takes_turns(cascade) || (frame as usize + cascade).is_multiple_of(sharing)
    }
}

impl Cascade {
    /// Whether all `corners` (view space) lie inside this cascade's shadow map volume.
    fn covers(&self, world_from_view: &Mat4, corners: &[Vec3A; 8], map_size: f32) -> bool {
        // Cascades snap to whole texels, so a slice may stick out by up to one.
        let limit = 1.0 + 2.0 / map_size;
        let clip_from_view = self.clip_from_world * *world_from_view;
        corners.iter().all(|corner| {
            let p = clip_from_view.project_point3a(*corner);
            p.x.abs() <= limit && p.y.abs() <= limit && (0.0..=1.0).contains(&p.z)
        })
    }

    /// Whether this cascade was drawn for a light with this rotation.
    fn has_light_rotation(&self, world_from_light: &Mat4) -> bool {
        let c = &self.world_from_cascade;
        c.x_axis == world_from_light.x_axis
            && c.y_axis == world_from_light.y_axis
            && c.z_axis == world_from_light.z_axis
    }
}

/// Sets up [`Cascades`] for all shadow mapped [`DirectionalLight`]s.
pub fn build_directional_light_cascades(
    directional_light_shadow_map: Res<DirectionalLightShadowMap>,
    views: Query<(Entity, &GlobalTransform, &Projection, &Camera)>,
    mut lights: Query<(
        &GlobalTransform,
        &DirectionalLight,
        &CascadeShadowConfig,
        &mut Cascades,
        Option<&CascadeReuse>,
    )>,
    mut frame: Local<u32>,
    mut last_map_size: Local<usize>,
) {
    *frame = frame.wrapping_add(1);
    // Cascades snap to whole texels of the map they were drawn into.
    let same_map_size = *last_map_size == directional_light_shadow_map.size;
    *last_map_size = directional_light_shadow_map.size;
    let views = views
        .iter()
        .filter_map(|(entity, transform, projection, camera)| {
            if camera.is_active {
                Some((entity, projection, transform.to_matrix()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for (transform, directional_light, cascades_config, mut cascades, reuse) in &mut lights {
        if !directional_light.shadow_maps_enabled {
            continue;
        }
        let previous = core::mem::take(&mut cascades.cascades);

        // It is very important to the numerical and thus visual stability of shadows that
        // `world_from_light` has orthogonal upper-left 3x3 and zero translation.
        // Even though only the direction (i.e. rotation) of the light matters, we don't constrain
        // users to not change any other aspects of the transform - there's no guarantee
        // `transform.to_matrix()` will give us a matrix with our desired properties.
        // Instead, we directly create a good matrix from just the rotation.
        let world_from_light = Mat4::from_quat(transform.rotation());
        // The transpose is the inverse for orthogonal matrices.
        let light_from_world = world_from_light.transpose();

        for (view_entity, projection, world_from_view) in views.iter().copied() {
            let light_view_from_camera = light_from_world * world_from_view;
            let overlap_factor = 1.0 - cascades_config.overlap_proportion;
            let far_bounds = cascades_config.bounds.iter();
            let near_bounds = [cascades_config.minimum_distance]
                .into_iter()
                .chain(far_bounds.clone().map(|bound| overlap_factor * bound));
            let map_size = directional_light_shadow_map.size as f32;
            let count = cascades_config.bounds.len();
            let view_cascades = near_bounds
                .zip(far_bounds)
                .enumerate()
                .map(|(index, (near_bound, far_bound))| {
                    // Negate bounds as -z is camera forward direction.
                    let corners = projection.get_frustum_corners(-near_bound, -far_bound);
                    let Some(reuse) = reuse.filter(|r| r.takes_turns(index)) else {
                        return calculate_cascade(
                            corners,
                            map_size,
                            world_from_light,
                            light_view_from_camera,
                            0.0,
                            0.0,
                        );
                    };
                    if same_map_size
                        && !reuse.is_due(index, count, *frame)
                        && let Some(old) = previous.get(&view_entity).and_then(|c| c.get(index))
                        && old.has_light_rotation(&world_from_light)
                        && old.covers(&world_from_view, &corners, map_size)
                    {
                        return Cascade {
                            reused: true,
                            ..old.clone()
                        };
                    }
                    calculate_cascade(
                        corners,
                        map_size,
                        world_from_light,
                        light_view_from_camera,
                        reuse.padding,
                        CascadeReuse::DEPTH_PADDING,
                    )
                })
                .collect();
            cascades.cascades.insert(view_entity, view_cascades);
        }
    }
}

/// Returns a [`Cascade`] for the frustum defined by `frustum_corners`.
///
/// The corner vertices should be specified in the following order:
/// first the bottom right, top right, top left, bottom left for the near plane, then similar for the far plane.
///
/// See this [reference](https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf) for more details.
fn calculate_cascade(
    frustum_corners: [Vec3A; 8],
    cascade_texture_size: f32,
    world_from_light: Mat4,
    light_from_camera: Mat4,
    padding: f32,
    depth_padding: f32,
) -> Cascade {
    let mut min = Vec3A::splat(f32::MAX);
    let mut max = Vec3A::splat(f32::MIN);
    for corner_camera_view in frustum_corners {
        let corner_light_view = light_from_camera.transform_point3a(corner_camera_view);
        min = min.min(corner_light_view);
        max = max.max(corner_light_view);
    }

    // NOTE: Use the larger of the frustum slice far plane diagonal and body diagonal lengths as this
    //       will be the maximum possible projection size. Use the ceiling to get an integer which is
    //       very important for floating point stability later. It is also important that these are
    //       calculated using the original camera space corner positions for floating point precision
    //       as even though the lengths using corner_light_view above should be the same, precision can
    //       introduce small but significant differences.
    // NOTE: The size remains the same unless the view frustum or cascade configuration is modified.
    let depth = depth_padding * (max.z - min.z);
    min.z -= depth;
    max.z += depth;
    let body_diagonal = (frustum_corners[0] - frustum_corners[6]).length_squared();
    let far_plane_diagonal = (frustum_corners[4] - frustum_corners[6]).length_squared();
    let cascade_diameter = (body_diagonal.max(far_plane_diagonal).sqrt() * (1.0 + padding)).ceil();

    // NOTE: If we ensure that cascade_texture_size is a power of 2, then as we made cascade_diameter an
    //       integer, cascade_texel_size is then an integer multiple of a power of 2 and can be
    //       exactly represented in a floating point value.
    let cascade_texel_size = cascade_diameter / cascade_texture_size;
    // NOTE: For shadow stability it is very important that the near_plane_center is at integer
    //       multiples of the texel size to be exactly representable in a floating point value.
    let near_plane_center = Vec3A::new(
        (0.5 * (min.x + max.x) / cascade_texel_size).floor() * cascade_texel_size,
        (0.5 * (min.y + max.y) / cascade_texel_size).floor() * cascade_texel_size,
        // NOTE: max.z is the near plane for right-handed y-up
        max.z,
    );

    // It is critical for `cascade_from_world` to be stable. So rather than forming `world_from_cascade`
    // and inverting it, which risks instability due to numerical precision, we directly form
    // `cascade_from_world` as the reference material suggests.
    let world_from_light_transpose = world_from_light.transpose();
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-near_plane_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_from_light * near_plane_center.extend(1.0),
    );

    // Right-handed orthographic projection, centered at `near_plane_center`.
    // NOTE: This is different from the reference material, as we use reverse Z.
    let r = (max.z - min.z).recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / cascade_diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / cascade_diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );

    let clip_from_world = clip_from_cascade * cascade_from_world;
    Cascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world,
        texel_size: cascade_texel_size,
        reused: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_camera::PerspectiveProjection;
    use bevy_transform::components::Transform;

    /// A world with one shadowed sun (4 cascades) and one camera; returns (world, sun, camera).
    fn scene(reuse: Option<CascadeReuse>) -> (World, Entity, Entity) {
        let mut world = World::new();
        world.insert_resource(DirectionalLightShadowMap { size: 2048 });
        let sun = world
            .spawn((
                GlobalTransform::from(
                    Transform::from_xyz(1.0, 2.0, 0.5)
                        .looking_at(bevy_math::Vec3::ZERO, bevy_math::Vec3::Y),
                ),
                DirectionalLight {
                    shadow_maps_enabled: true,
                    ..Default::default()
                },
                CascadeShadowConfigBuilder {
                    num_cascades: 4,
                    first_cascade_far_bound: 9.0,
                    maximum_distance: 540.0,
                    ..Default::default()
                }
                .build(),
                Cascades::default(),
            ))
            .id();
        if let Some(reuse) = reuse {
            world.entity_mut(sun).insert(reuse);
        }
        let camera = world
            .spawn((
                GlobalTransform::IDENTITY,
                Projection::Perspective(PerspectiveProjection::default()),
                Camera::default(),
            ))
            .id();
        (world, sun, camera)
    }

    fn frame(
        world: &mut World,
        schedule: &mut Schedule,
        sun: Entity,
        camera: Entity,
    ) -> Vec<Cascade> {
        schedule.run(world);
        world.get::<Cascades>(sun).unwrap().cascades[&camera].clone()
    }

    fn schedule() -> Schedule {
        let mut schedule = Schedule::default();
        schedule.add_systems(build_directional_light_cascades);
        schedule
    }

    fn reused(cascades: &[Cascade]) -> Vec<bool> {
        cascades.iter().map(|c| c.reused).collect()
    }

    #[test]
    fn without_the_component_every_cascade_redraws() {
        let (mut world, sun, camera) = scene(None);
        let mut s = schedule();
        for _ in 0..4 {
            assert_eq!(reused(&frame(&mut world, &mut s, sun, camera)), [false; 4]);
        }
    }

    #[test]
    fn far_cascades_take_turns_and_near_ones_always_redraw() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse::default()));
        let mut s = schedule();
        // Nothing to keep yet on the first frame.
        assert_eq!(reused(&frame(&mut world, &mut s, sun, camera)), [false; 4]);
        let a = reused(&frame(&mut world, &mut s, sun, camera));
        let b = reused(&frame(&mut world, &mut s, sun, camera));
        assert_eq!(&a[..2], &[false; 2]);
        assert_eq!(&b[..2], &[false; 2]);
        assert_ne!(a[2], a[3], "exactly one far cascade redraws per frame");
        assert_eq!((a[2], a[3]), (b[3], b[2]), "and they swap");
    }

    #[test]
    fn a_kept_cascade_keeps_the_matrix_it_was_drawn_with() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse {
            padding: 0.2,
            ..Default::default()
        }));
        let mut s = schedule();
        let first = frame(&mut world, &mut s, sun, camera);
        *world.get_mut::<GlobalTransform>(camera).unwrap() =
            Transform::from_xyz(0.5, 0.0, 0.0).into();
        let second = frame(&mut world, &mut s, sun, camera);
        let kept = (2..4)
            .find(|i| second[*i].reused)
            .expect("one far cascade is kept");
        assert_eq!(second[kept].clip_from_world, first[kept].clip_from_world);
        assert_ne!(second[0].clip_from_world, first[0].clip_from_world);
    }

    #[test]
    fn a_camera_cut_redraws_everything() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse {
            padding: 0.2,
            ..Default::default()
        }));
        let mut s = schedule();
        frame(&mut world, &mut s, sun, camera);
        *world.get_mut::<GlobalTransform>(camera).unwrap() =
            Transform::from_xyz(900.0, 0.0, 0.0).into();
        assert_eq!(reused(&frame(&mut world, &mut s, sun, camera)), [false; 4]);
    }

    #[test]
    fn a_moved_sun_redraws_everything() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse {
            padding: 0.2,
            ..Default::default()
        }));
        let mut s = schedule();
        frame(&mut world, &mut s, sun, camera);
        *world.get_mut::<GlobalTransform>(sun).unwrap() = Transform::from_xyz(1.0, 2.0, 0.502)
            .looking_at(bevy_math::Vec3::ZERO, bevy_math::Vec3::Y)
            .into();
        assert_eq!(reused(&frame(&mut world, &mut s, sun, camera)), [false; 4]);
    }

    #[test]
    fn a_resized_shadow_map_redraws_everything() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse::default()));
        let mut s = schedule();
        frame(&mut world, &mut s, sun, camera);
        world.resource_mut::<DirectionalLightShadowMap>().size = 4096;
        assert_eq!(reused(&frame(&mut world, &mut s, sun, camera)), [false; 4]);
    }

    #[test]
    fn two_cameras_keep_their_own_cascades() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse::default()));
        let other = world
            .spawn((
                GlobalTransform::from(Transform::from_xyz(300.0, 5.0, 0.0)),
                Projection::Perspective(PerspectiveProjection::default()),
                Camera::default(),
            ))
            .id();
        let mut s = schedule();
        frame(&mut world, &mut s, sun, camera);
        let first = frame(&mut world, &mut s, sun, camera);
        let second = world.get::<Cascades>(sun).unwrap().cascades[&other].clone();
        assert_eq!(reused(&first), reused(&second));
        assert_ne!(first[2].clip_from_world, second[2].clip_from_world);
    }

    #[test]
    fn the_view_slice_always_lies_inside_its_cascade() {
        let (mut world, sun, camera) = scene(Some(CascadeReuse::default()));
        let mut s = schedule();
        let bounds = world.get::<CascadeShadowConfig>(sun).unwrap().clone();
        for step in 0..60 {
            let x = step as f32 * 2.3;
            let transform = Transform::from_xyz(x, 10.0, -x)
                .with_rotation(bevy_math::Quat::from_rotation_y(step as f32 * 0.03));
            *world.get_mut::<GlobalTransform>(camera).unwrap() = transform.into();
            let cascades = frame(&mut world, &mut s, sun, camera);
            let projection = world.get::<Projection>(camera).unwrap().clone();
            let mut near = bounds.minimum_distance;
            for (cascade, far) in cascades.iter().zip(&bounds.bounds) {
                let corners = projection.get_frustum_corners(-near, -far);
                for corner in corners {
                    let p = cascade
                        .clip_from_world
                        .project_point3a(transform.to_matrix().transform_point3a(corner));
                    assert!(
                        p.x.abs() <= 1.001 && p.y.abs() <= 1.001 && (-0.001..=1.001).contains(&p.z),
                        "step {step} far {far}: {p:?}"
                    );
                }
                near = (1.0 - bounds.overlap_proportion) * far;
            }
        }
    }
}
