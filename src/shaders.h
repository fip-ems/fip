#pragma once

static const char* vertShader = R"(
#version 330 core
in vec2 terrain;
out vec2 TexCoord;
out vec3 FragPos;
out vec2 TerrainPos;
out float Valid;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int W;
uniform int H;
uniform sampler2D texWater;
uniform sampler2D texTerrain;
uniform float terrainScale;
uniform float invalidTerrain;
void main()
{
	float dx = 1.0;
	float x = terrain.x;
	float z = terrain.y;
	float u = x/W;
	float v = z/H;
	TexCoord = vec2(u, v);
	float sohle = texture(texTerrain, TexCoord).r;
	float ah = texture(texWater, TexCoord).r;
	vec4 pos = model * vec4(x, terrainScale * (ah+sohle), H-z, 1.0);
	gl_Position = projection * view * pos;
	FragPos = pos.xyz;
	TerrainPos = vec2(x, z);
	Valid = (sohle != invalidTerrain) ? 1.f : 0.f;
}
)";


static const char* fragShader = R"(
#version 330 core
in vec2 TexCoord;
in vec3 FragPos;
in vec2 TerrainPos;
in float Valid;
uniform sampler2D texWater;
uniform sampler2D texTerrain;
uniform sampler2D texQx;
uniform sampler2D texQy;
uniform sampler2D texColor;
uniform vec3 viewPos;
uniform int vis;
uniform bool hasTexture;
uniform ivec2 highlight_xy;
uniform float maxDepth;
uniform float maxVelocity;

// https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/MATLAB_jet.frag
float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}
float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}
float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}
vec4 colormap_jet(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}
void main()
{
	float ah = texture(texWater, TexCoord).r;
	float sohle = texture(texTerrain, TexCoord).r;
	//if (mod(TerrainPos.x, 26.f) < 0.5 || mod(TerrainPos.y, 50.f) < 0.5)
	//	gl_FragColor = vec4(0,0,0,1);
	//else if (ah > 0.01){
	if (Valid < 0.9999f) {
		gl_FragColor = vec4(0, 0, 0, 0);
	}
	else if (ah > 0.01){
		if (vis == 0) {
			gl_FragColor = colormap_jet(min(1.f, ah / maxDepth));
		}
		if (vis == 1) {
			float ah_east = textureOffset(texWater, TexCoord, ivec2(1, 0)).r;
			float ah_north = textureOffset(texWater, TexCoord, ivec2(0, 1)).r;

			float qx = texture(texQx, TexCoord).r / max(0.01, (ah + ah_east) * 0.5f);
			float qy = texture(texQy, TexCoord).r / max(0.01, (ah + ah_north) * 0.5f);
			float v = sqrt(qx * qx + qy * qy);// / ah;
			gl_FragColor = colormap_jet(min(1.f, v / maxVelocity));
		}
	}
	else {
		float hL = textureOffset(texTerrain, TexCoord, ivec2(-1, 0)).r;
		float hR = textureOffset(texTerrain, TexCoord, ivec2(1, 0)).r;
		float hD = textureOffset(texTerrain, TexCoord, ivec2(0, -1)).r;
		float hU = textureOffset(texTerrain, TexCoord, ivec2(0, 1)).r;
		vec3 normal = normalize(vec3(hL - hR, 2.0, hD - hU));
		vec3 lightDir = normalize(vec3(1, -5, 1));
		vec3 reflectDir = reflect(lightDir, normal);
		vec3 viewDir = normalize(viewPos - FragPos);
		vec3 color = vec3(1, 1, 1);
		if (hasTexture)
			color = texture(texColor, TexCoord).rgb;
		float ambient = 0.15;
		float diffuse = max(0, dot(normal, -lightDir));
		float specular = pow(max(0, dot(viewDir, reflectDir)), 4);
		gl_FragColor = vec4(color * (ambient + 0.55 * diffuse + 0.3 * specular), 1);
	}
	//if (TerrainPos.x >= 1360 && TerrainPos.x <= 1370 && TerrainPos.y >= 4660 && TerrainPos.y <= 4671)
	//	gl_FragColor = vec4(1, 0, 0, 1);
	//if (TerrainPos.x >= highlight_xy.x && TerrainPos.x < highlight_xy.x + 28 &&
	//	TerrainPos.y >= highlight_xy.y && TerrainPos.y < highlight_xy.y + 50)
	//	gl_FragColor += vec4(1, 0, 1, 0.75);
}
)";
