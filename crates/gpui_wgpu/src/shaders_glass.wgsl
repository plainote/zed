// Glass (Vesper L0-10/L0-14 spike), the WGSL port of the Metal pass. The
// renderer blurs everything drawn before a glass batch into a half-resolution
// dual-Kawase pyramid, then draws each glass shape by sampling the blur through
// a lens that bends its edge band. Concatenated after `shaders.wgsl` and
// `shaders_storage.wgsl`; the blur passes bind only `t_sprite` and `s_sprite`.

struct Glass {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    corner_radius: f32,
    bezel: f32,
    refraction: f32,
    rim: f32,
    tint: Hsla,
    dispersion: f32,
    // 0 samples the first, lightly blurred pyramid level (t_light), 1 the full blur.
    frost: f32,
    pad2: vec2<u32>,
}

@group(1) @binding(0) var<storage, read> b_glass: array<Glass>;
// The first pyramid level, for clear (unfrosted) glass. Only fs_glass uses it.
@group(3) @binding(0) var t_light: texture_2d<f32>;
@group(3) @binding(1) var s_light: sampler;

struct GlassBlurVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the whole target.
@vertex
fn vs_glass_blur(@builtin(vertex_index) vertex_id: u32) -> GlassBlurVarying {
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    var out = GlassBlurVarying();
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// The target is half the source, so half a target texel is one source texel.
@fragment
fn fs_glass_downsample(input: GlassBlurVarying) -> @location(0) vec4<f32> {
    let o = 1.0 / vec2<f32>(textureDimensions(t_sprite, 0));
    var sum = textureSampleLevel(t_sprite, s_sprite, input.uv, 0.0) * 4.0;
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv - o, 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + o, 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(o.x, -o.y), 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv - vec2<f32>(o.x, -o.y), 0.0);
    return sum / 8.0;
}

// The target is twice the source, so half a target texel is a quarter source texel.
@fragment
fn fs_glass_upsample(input: GlassBlurVarying) -> @location(0) vec4<f32> {
    let o = 0.25 / vec2<f32>(textureDimensions(t_sprite, 0));
    var sum = textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(-o.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(-o.x, o.y), 0.0) * 2.0;
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(0.0, o.y * 2.0), 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(o.x, o.y), 0.0) * 2.0;
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(o.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(o.x, -o.y), 0.0) * 2.0;
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(0.0, -o.y * 2.0), 0.0);
    sum += textureSampleLevel(t_sprite, s_sprite, input.uv + vec2<f32>(-o.x, -o.y), 0.0) * 2.0;
    return sum / 12.0;
}

struct GlassVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) glass_id: u32,
    @location(1) clip_distances: vec4<f32>,
}

@vertex
fn vs_glass(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> GlassVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let g = b_glass[instance_id];
    var out = GlassVarying();
    out.position = to_device_position(unit_vertex, g.bounds);
    out.glass_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, g.bounds, g.content_mask);
    return out;
}

@fragment
fn fs_glass(input: GlassVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let g = b_glass[input.glass_id];

    let p = input.position.xy;
    let half_size = g.bounds.size * 0.5;
    let center = g.bounds.origin + half_size;
    let radius = min(g.corner_radius, min(half_size.x, half_size.y));
    let rel = p - center;
    let q = abs(rel) - (half_size - radius);
    // Signed distance to the rounded rectangle; negative inside.
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    let coverage = saturate(0.5 - d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    let tint = hsla_to_rgba(g.tint);
    // A 1x1 blur texture means this surface can't be sampled: flat tint only.
    let blur_size = vec2<f32>(textureDimensions(t_sprite, 0));
    if (blur_size.x <= 1.0) {
        return blend_color(tint, coverage);
    }

    // Outward normal of the nearest edge.
    var n = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), q.x > q.y);
    if (q.x > 0.0 && q.y > 0.0) {
        n = normalize(q);
    }
    n *= select(vec2<f32>(-1.0), vec2<f32>(1.0), rel >= vec2<f32>(0.0));

    // 1 on the rim, 0 once past the bezel; a circular profile, like a lens edge.
    let edge = saturate(1.0 + d / max(g.bezel, 1.0));
    let bend = 1.0 - sqrt(saturate(1.0 - edge * edge));

    let viewport = globals.viewport_size;
    let offset = -n * bend * g.refraction;
    let uv = (p + offset) / viewport;
    // A little dispersion along the edge.
    let dispersion = -n * bend * g.dispersion * g.refraction / viewport;
    let frosted = vec3<f32>(
        textureSampleLevel(t_sprite, s_sprite, uv + dispersion, 0.0).r,
        textureSampleLevel(t_sprite, s_sprite, uv, 0.0).g,
        textureSampleLevel(t_sprite, s_sprite, uv - dispersion, 0.0).b,
    );
    let clear = vec3<f32>(
        textureSampleLevel(t_light, s_light, uv + dispersion, 0.0).r,
        textureSampleLevel(t_light, s_light, uv, 0.0).g,
        textureSampleLevel(t_light, s_light, uv - dispersion, 0.0).b,
    );
    var color = mix(clear, frosted, clamp(g.frost, 0.0, 1.0));

    color = mix(color, tint.rgb, tint.a);

    // Rim light from the top left: a hairline plus a soft glow inside the bezel.
    let facing = dot(n, normalize(vec2<f32>(-0.45, -1.0)));
    let hairline = saturate(1.0 + d / 1.5);
    let glow = edge * edge * edge;
    color += g.rim * (hairline * (0.25 + 0.5 * abs(facing)) + glow * 0.18 * max(facing, 0.0));

    return blend_color(vec4<f32>(saturate(color), 1.0), coverage);
}
