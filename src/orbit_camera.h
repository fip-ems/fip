#pragma once
#include "flythrough_camera.h"
#include <cmath>

#define M_PI 3.14159265359

void orbitcam_update(
    float eye[3],
    float pivot[3],
    const float up[3],
    float dir_out[3],
    float view[16],
    float delta_time_seconds,
    float eye_speed,
    float degrees_per_cursor_move,
    int delta_cursor_x, int delta_cursor_y, double delta_scroll,
    int lmb, int rmb
)
{
    auto cross = [](const float a[3], const float b[3], float res[3]){
        res[0] = a[1] * b[2] - a[2] * b[1];
        res[1] = a[2] * b[0] - a[0] * b[2];
        res[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto magnitude = [](const float a[3]) {
        return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    };
    auto normalize = [magnitude](float a[3]) {
        float len = magnitude(a);
        a[0] /= len;
        a[1] /= len;
        a[2] /= len;
    };
    auto normalize2 = [](float a[2]) {
        float len = sqrtf(a[0] * a[0] + a[1] * a[1]);
        a[0] /= len;
        a[1] /= len;
    };

    float dir[3] = { eye[0] - pivot[0], eye[1] - pivot[1], eye[2] - pivot[2] };
    float radius = magnitude(dir);
    normalize(dir);
    // cross product between up and dir
    float eye_right[3]; cross(dir, up, eye_right); normalize(eye_right);
    // cross product between right and dir
    float eye_up[3]; cross(eye_right, dir, eye_up);
    float azimuth = atan2f(dir[2], dir[0]);
    float polar = atan2f(dir[1], sqrtf(dir[0] * dir[0] + dir[2] * dir[2]));

    if (lmb) {
        // rotate azimuth
        azimuth += degrees_per_cursor_move * delta_cursor_x * delta_time_seconds;
        azimuth = fmodf(azimuth, 2 * M_PI);
        if (azimuth < 0.f)
            azimuth += 2 * M_PI;

        // rotate polar
        polar += degrees_per_cursor_move * delta_cursor_y * delta_time_seconds;
        const float polarCap = M_PI / 2.f - 0.001f;
        polar = fminf(polarCap, fmaxf(0.f, polar));
    }

    if (rmb) {
        float up2[2] = { eye_up[0], eye_up[2] }; normalize2(up2);
        pivot[0] += (up2[0] * delta_cursor_y + eye_right[0] * delta_cursor_x) * eye_speed * delta_time_seconds * radius * 0.1f;
        //pivot[1] += eye_up[1] * delta_cursor_y * eye_speed * delta_time_seconds * sqrtf(radius);
        pivot[2] += (up2[1] * delta_cursor_y + eye_right[2] * delta_cursor_x) * eye_speed * delta_time_seconds * radius * 0.1f;
    }

    radius -= delta_scroll * radius * 0.1f;
    if (radius < 1.f)
        radius = 1.f;

    const float sa = sinf(azimuth);
    const float ca = cosf(azimuth);
    const float sp = sinf(polar);
    const float cp = cosf(polar);

    eye[0] = pivot[0] + radius * cp * ca;
    eye[1] = pivot[1] + radius * sp;
    eye[2] = pivot[2] + radius * cp * sa;

    dir_out[0] = { pivot[0] - eye[0] };
    dir_out[1] = { pivot[1] - eye[1] };
    dir_out[2] = { pivot[2] - eye[2] };

    flythrough_camera_look_to(eye, dir_out, up, view, 0);
}
