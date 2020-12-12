#version 150

in vec3 position;
in vec3 normal;

out vec3 viewPosition;
out vec3 viewNormal;

uniform mat4 modelViewMatrix;
uniform mat4 normalMatrix;
uniform mat4 projectionMatrix;

void main() {
  vec4 viewPos = modelViewMatrix * vec4(position, 1.0f);
  viewPosition = viewPos.xyz;

  gl_Position = projectionMatrix * viewPos;
  viewNormal = normalize((normalMatrix*vec4(normal, 0.0f)).xyz);
}

