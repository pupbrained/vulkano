#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_coord;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

// Adjust light direction and add ambient light
const vec3 LIGHT = vec3(1.0, 1.0, 1.0);
const float AMBIENT_STRENGTH = 0.3;

void main() {
  vec4 tex_color = texture(tex_sampler, v_tex_coord);

  // Calculate diffuse lighting
  float diffuse = max(dot(normalize(v_normal), normalize(LIGHT)), 0.0);

  // Add ambient light and increase overall brightness
  vec3 ambient = AMBIENT_STRENGTH * tex_color.rgb;
  vec3 result = ambient + diffuse * tex_color.rgb;

  // Apply gamma correction to make it appear brighter
  result = pow(result, vec3(1.0/2.2));

  f_color = vec4(result, tex_color.a);
}
