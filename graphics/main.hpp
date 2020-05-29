#pragma once
#include <math.h>
#define pi (3.14159265359f)
typedef unsigned char u8;

struct vec3
{
    float x;
    float y;
    float z;
};

inline
vec3 operator-(vec3 v)
{
    return {-v.x, -v.y, -v.z};
}

inline
vec3 operator+(vec3 lhs, vec3 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline
vec3 operator-(vec3 lhs, vec3 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline
vec3 operator*(float scalar, vec3 rhs)
{
    return {scalar * rhs.x, scalar * rhs.y, scalar * rhs.z};
}

inline
vec3 mul_cwise(vec3 lhs, vec3 rhs)
{
    return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

struct vec4
{
    float x;
    float y;
    float z;
    float w;
};

inline
vec4 operator-(vec4 v)
{
    return {-v.x, -v.y, -v.z, -v.w};
}

inline
vec4 operator+(vec4 lhs, vec4 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
}

inline
vec4 operator-(vec4 lhs, vec4 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
}

inline
vec4 operator*(float scalar, vec4 rhs)
{
    return {scalar * rhs.x, scalar * rhs.y, scalar * rhs.z, scalar * rhs.w};
}

// matrix functions assume row based matrix storage (row1, row2, row3, ...)
// vec3 and vec4 are treated as column vectors in matrix vector multiplication

struct mat3
{
    float data[9];
};

struct mat4
{
    float data[16];
};

inline
float dot(vec3 lhs, vec3 rhs)
{
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

inline
float dot(vec4 lhs, vec4 rhs)
{
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
}

inline
vec3 cross(vec3 lhs, vec3 rhs)
{
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
}

inline
float length(vec3 v)
{
    return sqrtf(dot(v, v));
}

inline
vec3 normalize(vec3 v)
{
    float l = length(v);
    return {v.x / l, v.y / l, v.z / l};
}

inline
mat3 add(mat3 lhs, mat3 rhs)
{
    for(int i = 0; i < 9; ++i)
        lhs.data[i] += rhs.data[i];
    return lhs;
}

inline
mat4 add(mat4 lhs, mat4 rhs)
{
    for(int i = 0; i < 16; ++i)
        lhs.data[i] += rhs.data[i];
    return lhs;
}

inline
vec3 mul(mat3 lhs, vec3 rhs)
{
    vec3 product;

    for(int i = 0; i < 3; ++i)
        (&product.x)[i] = dot(vec3{lhs.data[i*3], lhs.data[i*3 + 1], lhs.data[i*3 + 2]}, rhs);
    return product;
}

inline
vec4 mul(mat4 lhs, vec4 rhs)
{
    vec4 product;

    for(int i = 0; i < 4; ++i)
        (&product.x)[i] = dot(vec4{lhs.data[i*4], lhs.data[i*4 + 1], lhs.data[i*4 + 2], lhs.data[i*4 + 3]}, rhs);
    return product;
}

inline
mat3 mul(mat3 lhs, mat3 rhs)
{
    for(int y = 0; y < 3; ++y)
    {
        vec3 column = {rhs.data[y], rhs.data[y + 3], rhs.data[y + 6]};
        column = mul(lhs, column);
        rhs.data[y] = column.x;
        rhs.data[y + 3] = column.y;
        rhs.data[y + 6] = column.z;
    }
    return rhs;
}

inline
mat4 mul(mat4 lhs, mat4 rhs)
{
    for(int y = 0; y < 4; ++y)
    {
        vec4 column = {rhs.data[y], rhs.data[y + 4], rhs.data[y + 8], rhs.data[y + 12]};
        column = mul(lhs, column);
        rhs.data[y] = column.x;
        rhs.data[y + 4] = column.y;
        rhs.data[y + 8] = column.z;
        rhs.data[y + 12] = column.w;
    }
    return rhs;
}

inline
mat3 transpose(mat3 lhs)
{
    mat3 t;

    for(int i = 0; i < 3; ++i)
    {
        t.data[i] = lhs.data[i * 3];
        t.data[i + 3] = lhs.data[i * 3 + 1];
        t.data[i + 6] = lhs.data[i * 3 + 2];
    }
    return t;
}

inline
mat4 transpose(mat4 lhs)
{
    mat4 t;

    for(int i = 0; i < 4; ++i)
    {
        t.data[i] = lhs.data[i * 4];
        t.data[i + 4] = lhs.data[i * 4 + 1];
        t.data[i + 8] = lhs.data[i * 4 + 2];
        t.data[i + 12] = lhs.data[i * 4 + 3];
    }
    return t;
}

inline
mat3 mat4_to_mat3(mat4 m4)
{
    mat3 m3;

    for(int i = 0; i < 3; ++i)
    {
        m3.data[i*3 + 0] = m4.data[i*4 + 0];
        m3.data[i*3 + 1] = m4.data[i*4 + 1];
        m3.data[i*3 + 2] = m4.data[i*4 + 2];
    }
    return m3;
}

inline
float deg_to_rad(float d)
{
    return 2.f * pi * d / 360.f;
}

inline
mat3 identity3()
{
    mat3 m = {};
    m.data[0] = 1;
    m.data[5] = 1;
    m.data[8] = 1;
    return m;
}

inline
mat4 identity4()
{
    mat4 m = {};
    m.data[0] = 1;
    m.data[5] = 1;
    m.data[10] = 1;
    m.data[15] = 1;
    return m;
}

inline
mat4 translate(vec3 v)
{
    mat4 m = {};
    m.data[0] = 1;
    m.data[3] = v.x;
    m.data[5] = 1;
    m.data[7] = v.y;
    m.data[10] = 1;
    m.data[11] = v.z;
    m.data[15] = 1;
    return m;
}

inline
mat4 scale(vec3 v)
{
    mat4 m = {};
    m.data[0] = v.x;
    m.data[5] = v.y;
    m.data[11] = v.z;
    m.data[15] = 1;
    return m;
}

// angle is in radians

inline
mat4 rotate_x(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 m = {};
    m.data[0] = 1;
    m.data[5] = c;
    m.data[6] = -s;
    m.data[9] = s;
    m.data[10] = c;
    m.data[15] = 1;
    return m;
}

inline
mat4 rotate_y(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 m = {};
    m.data[0] = c;
    m.data[2] = s;
    m.data[5] = 1;
    m.data[8] = -s;
    m.data[10] = c;
    m.data[15] = 1;
    return m;
}

inline
mat4 rotate_z(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 m = {};
    m.data[0] = c;
    m.data[1] = -s;
    m.data[4] = s;
    m.data[5] = c;
    m.data[10] = 1;
    m.data[15] = 1;
    return m;
}

inline
mat4 rotate_axis(vec3 a, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4 m = {};
    m.data[0] = c + ((1-c) * a.x * a.x);
    m.data[1] = ((1-c) * a.x * a.y) - (s * a.z);
    m.data[2] = ((1-c) * a.x * a.z) + (s * a.y);
    m.data[4] = ((1-c) * a.x * a.y) + (s * a.z);
    m.data[5] = c + ((1-c) * a.y * a.y);
    m.data[6] = ((1-c) * a.y * a.z) - (s * a.x);
    m.data[8] = ((1-c) * a.x * a.z) - (s * a.y);
    m.data[9] = ((1-c) * a.y * a.z) + (s * a.x);
    m.data[10] = c + ((1-c) * a.z * a.z);
    m.data[15] = 1;
    return m;
}

// transforms from world coordinates to camera coordinates

inline
mat4 lookat(vec3 pos, vec3 dir)
{
    // camera basis vectors with respect to the world coordinate system
    vec3 x = normalize(cross(dir, vec3{0, 1, 0}));
    vec3 y = cross(x, dir);
    vec3 z = -dir;
    // inverted change of basis (from camera to world)
    mat4 m = {};
    m.data[0] = x.x;
    m.data[1] = x.y;
    m.data[2] = x.z;
    m.data[4] = y.x;
    m.data[5] = y.y;
    m.data[6] = y.z;
    m.data[8] = z.x;
    m.data[9] = z.y;
    m.data[10] = z.z;
    m.data[15] = 1;
    return mul(m, translate(-pos)); // first translate then rotate
}

// angles are in degrees

inline
mat4 lookat(vec3 pos, float yaw, float pitch)
{
    yaw = deg_to_rad(yaw);
    pitch = deg_to_rad(pitch);
    mat4 m = mul(rotate_y(-yaw), rotate_x(-pitch));
    return mul(m, translate(-pos));
}

inline
mat4 frustum(float l, float r, float b, float t, float n, float f)
{
    mat4 m = {};
    m.data[0] = 2*n/(r-l);
    m.data[2] = (r+l)/(r-l);
    m.data[5] = 2*n/(t-b);
    m.data[6] = (t+b)/(t-b);
    m.data[10] = -(f+n)/(f-n);
    m.data[11] = -(2*n*f)/(f-n);
    m.data[14] = -1;
    return m;
}

// fovy is in degrees

inline
mat4 perspective(float fovy, float aspect, float near, float far)
{
    float top = tanf(deg_to_rad(fovy) / 2.f) * near;
    float right = top * aspect;
    return frustum(-right, right, -top, top, near, far);
}

inline
mat4 orthographic(float l, float r, float b, float t, float n, float f)
{
    mat4 m = {};
    m.data[0] = 2/(r-l);
    m.data[3] = -(r+l)/(r-l);
    m.data[5] = 2/(t-b);
    m.data[7] = -(t+b)/(t-b);
    m.data[10] = -2/(f-n);
    m.data[11] = -(f+n)/(f-n);
    m.data[15] = 1;
    return m;
}

inline
mat3 inverse(mat3 m)
{
    mat3 inv;
    inv.data[0] = m.data[4]*m.data[8] - m.data[5]*m.data[7];
    inv.data[1] = m.data[2]*m.data[7] - m.data[1]*m.data[8];
    inv.data[2] = m.data[1]*m.data[5] - m.data[2]*m.data[4];
    inv.data[3] = m.data[5]*m.data[6] - m.data[3]*m.data[8];
    inv.data[4] = m.data[0]*m.data[8] - m.data[2]*m.data[6];
    inv.data[5] = m.data[2]*m.data[3] - m.data[0]*m.data[5];
    inv.data[6] = m.data[3]*m.data[7] - m.data[4]*m.data[6];
    inv.data[7] = m.data[1]*m.data[6] - m.data[0]*m.data[7];
    inv.data[8] = m.data[0]*m.data[4] - m.data[1]*m.data[3];

    float det = (m.data[0]*inv.data[0]) + (m.data[1]*inv.data[3]) + (m.data[2]*inv.data[6]);

    for(int i = 0; i < 9; ++i)
        inv.data[i] /= det;
    return inv;
}

inline
vec4 quat_mul(vec4 q1, vec4 q2)
{
    vec4 p;
    p.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    p.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    p.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    p.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    return p;
}

inline
vec4 quat_rot(vec3 axis, float angle)
{
    float s = sinf(angle / 2);
    vec4 q;
    q.w = cosf(angle / 2);
    q.x = s * axis.x;
    q.y = s * axis.y;
    q.z = s * axis.z;
    return q;
}

inline
mat4 quat_to_mat4(vec4 q)
{
    mat4 m = {};
    m.data[0] = 1 - (2 * q.y * q.y) - (2 * q.z * q.z);
    m.data[1] = (2 * q.x * q.y) - (2 * q.w * q.z);
    m.data[2] = (2 * q.x * q.z) + (2 * q.w * q.y);
    m.data[4] = (2 * q.x * q.y) + (2 * q.w * q.z);
    m.data[5] = 1 - (2 * q.x * q.x) - (2* q.z * q.z);
    m.data[6] = (2 * q.y * q.z) - (2 * q.w * q.x);
    m.data[8] = (2 * q.x * q.z) - (2 * q.w * q.y);
    m.data[9] = (2 * q.y * q.z) + (2 * q.w * q.x);
    m.data[10] = 1 - (2 * q.x * q.x) - (2 * q.y * q.y);
    m.data[15] = 1;
    return m;
}
