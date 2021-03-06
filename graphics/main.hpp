#pragma once
#include <math.h>
#define pi (3.14159265359f)
typedef unsigned char u8;

// note: angles are in radians

inline
float max(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

inline
float min(float lhs, float rhs)
{
    return lhs < rhs ? lhs : rhs;
}

struct vec2
{
    float x;
    float y;

    float& operator[](int idx)
    {
        return (&x)[idx];
    }
};

inline
vec2 operator-(vec2 v)
{
    return {-v.x, -v.y};
}

inline
vec2 operator+(vec2 lhs, vec2 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

inline
vec2 operator-(vec2 lhs, vec2 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}

inline
vec2 operator*(float scalar, vec2 rhs)
{
    return {scalar * rhs.x, scalar * rhs.y};
}

struct vec3
{
    float x;
    float y;
    float z;

    float& operator[](int idx)
    {
        return (&x)[idx];
    }
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

    float& operator[](int idx)
    {
        return (&x)[idx];
    }
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
// vec3 and vec4 are treated as column vectors in a matrix vector multiplication

struct mat3
{
    float data[9];

    vec3 row(int idx)
    {
        vec3 v;
        for(int i = 0; i < 3; ++i)
            v[i] = data[idx*3 + i];
        return v;
    }

    vec3 col(int idx)
    {
        vec3 v;
        for(int i = 0; i < 3; ++i)
            v[i] = data[i*3 + idx];
        return v;
    }
};

struct mat4
{
    float data[16];

    vec3 row(int idx)
    {
        vec3 v;
        for(int i = 0; i < 3; ++i)
            v[i] = data[idx*4 + i];
        return v;
    }

    vec3 col(int idx)
    {
        vec3 v;
        for(int i = 0; i < 3; ++i)
            v[i] = data[i*4 + idx];
        return v;
    }
};

inline
float dot(vec2 lhs, vec2 rhs)
{
    return (lhs.x * rhs.x) + (lhs.y * rhs.y);
}

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
float length(vec2 v)
{
    return sqrtf(dot(v, v));
}

inline
float length(vec3 v)
{
    return sqrtf(dot(v, v));
}

inline
float length(vec4 v)
{
    return sqrtf(dot(v, v));
}

inline
vec2 normalize(vec2 v)
{
    float l = length(v);
    return {v.x / l, v.y / l};
}

inline
vec3 normalize(vec3 v)
{
    float l = length(v);
    return {v.x / l, v.y / l, v.z / l};
}

inline
vec4 normalize(vec4 v)
{
    float l = length(v);
    return {v.x / l, v.y / l, v.z / l, v.w / l};
}

inline
vec3 operator*(mat3 lhs, vec3 rhs)
{
    vec3 product;

    for(int i = 0; i < 3; ++i)
        (&product.x)[i] = dot(vec3{lhs.data[i*3], lhs.data[i*3 + 1], lhs.data[i*3 + 2]}, rhs);
    return product;
}

inline
vec4 operator*(mat4 lhs, vec4 rhs)
{
    vec4 product;

    for(int i = 0; i < 4; ++i)
        (&product.x)[i] = dot(vec4{lhs.data[i*4], lhs.data[i*4 + 1], lhs.data[i*4 + 2], lhs.data[i*4 + 3]}, rhs);
    return product;
}

inline
mat3 operator*(mat3 lhs, mat3 rhs)
{
    for(int y = 0; y < 3; ++y)
    {
        vec3 column = {rhs.data[y], rhs.data[y + 3], rhs.data[y + 6]};
        column = lhs * column;
        rhs.data[y] = column.x;
        rhs.data[y + 3] = column.y;
        rhs.data[y + 6] = column.z;
    }
    return rhs;
}

inline
mat4 operator*(mat4 lhs, mat4 rhs)
{
    for(int y = 0; y < 4; ++y)
    {
        vec4 column = {rhs.data[y], rhs.data[y + 4], rhs.data[y + 8], rhs.data[y + 12]};
        column = lhs * column;
        rhs.data[y] = column.x;
        rhs.data[y + 4] = column.y;
        rhs.data[y + 8] = column.z;
        rhs.data[y + 12] = column.w;
    }
    return rhs;
}

inline
mat3 operator+(mat3 lhs, mat3 rhs)
{
    for(int i = 0; i < 9; ++i)
        lhs.data[i] += rhs.data[i];
    return lhs;
}

inline
mat4 operator+(mat4 lhs, mat4 rhs)
{
    for(int i = 0; i < 16; ++i)
        lhs.data[i] += rhs.data[i];
    return lhs;
}

inline
mat3 operator*(float sc, mat3 mat)
{
    for(int i = 0; i < 9; ++i)
        mat.data[i] *= sc;
    return mat;
}

inline
mat4 operator*(float sc, mat4 mat)
{
    for(int i = 0; i < 16; ++i)
        mat.data[i] *= sc;
    return mat;
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
mat3 to_mat3(mat4 m4)
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
mat4 to_mat4(mat3 m3)
{
    mat4 m4 = {};
    m4.data[15] = 1;

    for(int i = 0; i < 3; ++i)
    {
        m4.data[i*4 + 0] = m3.data[i*3 + 0];
        m4.data[i*4 + 1] = m3.data[i*3 + 1];
        m4.data[i*4 + 2] = m3.data[i*3 + 2];
    }
    return m4;
}

inline
vec4 to_point4(vec3 v)
{
    return {v.x, v.y, v.z, 1};
}

inline
vec4 to_dir4(vec3 v)
{
    return {v.x, v.y, v.z, 0};
}

inline
vec3 to_point3(vec2 v)
{
    return {v.x, v.y, 1};
}

inline
vec3 to_dir3(vec2 v)
{
    return {v.x, v.y, 0};
}

inline
vec3 to_vec3(vec4 v)
{
    return {v.x, v.y, v.z};
}

inline
vec2 to_vec2(vec3 v)
{
    return {v.x, v.y};
}

inline
float to_radians(float angle)
{
    return 2.f * pi * angle / 360.f;
}

inline
mat3 identity3()
{
    mat3 m = {};
    m.data[0] = 1;
    m.data[4] = 1;
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
    m.data[10] = v.z;
    m.data[15] = 1;
    return m;
}

inline
mat3 rotate_x(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat3 m = {};
    m.data[0] = 1;
    m.data[4] = c;
    m.data[5] = -s;
    m.data[7] = s;
    m.data[8] = c;
    return m;
}

inline
mat3 rotate_y(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat3 m = {};
    m.data[0] = c;
    m.data[2] = s;
    m.data[4] = 1;
    m.data[6] = -s;
    m.data[8] = c;
    return m;
}

inline
mat3 rotate_z(float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat3 m = {};
    m.data[0] = c;
    m.data[1] = -s;
    m.data[3] = s;
    m.data[4] = c;
    m.data[8] = 1;
    return m;
}

inline
mat3 rotate_axis(vec3 a, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat3 m;
    m.data[0] = c + ((1-c) * a.x * a.x);
    m.data[1] = ((1-c) * a.x * a.y) - (s * a.z);
    m.data[2] = ((1-c) * a.x * a.z) + (s * a.y);
    m.data[3] = ((1-c) * a.x * a.y) + (s * a.z);
    m.data[4] = c + ((1-c) * a.y * a.y);
    m.data[5] = ((1-c) * a.y * a.z) - (s * a.x);
    m.data[6] = ((1-c) * a.x * a.z) - (s * a.y);
    m.data[7] = ((1-c) * a.y * a.z) + (s * a.x);
    m.data[8] = c + ((1-c) * a.z * a.z);
    return m;
}

inline
mat4 rotate_x4(float angle)
{
    return to_mat4(rotate_x(angle));
}

inline
mat4 rotate_y4(float angle)
{
    return to_mat4(rotate_y(angle));
}

inline
mat4 rotate_z4(float angle)
{
    return to_mat4(rotate_z(angle));
}

inline
mat4 rotate_axis4(vec3 a, float angle)
{
    return to_mat4(rotate_axis(a, angle));
}

// notes on changing coordinate systems

// if we have basis vectors of system B expressed in system A and the origin of B (with respect to A) is equal to
// the origin of A translated by a vector T, then:
// [V]A = ( [B1, B2, B3] * [V]B ) + T
// which can be written in a more compact way:
// [V]A = M * [V]B
// M = | B1x B2x B3x Tx |
//     | B1y B2y B3y Ty |
//     | B1z B2z B3z Tz |
//     |   0   0   0  1 |
// V is first rotated and then translated;
// inverse of M transforms in the other direction, from A to B;
// to calculate the inverse we don't need to use the expensive inverse() function because of the orthogonality
// of the rotation part of M, see invert_coord_change()

// e.g. transforms (model to view) matrix to (view to model)
inline
mat4 invert_coord_change(mat4 m)
{
    mat3 rot_trans = transpose(to_mat3(m));
    vec3 tr = {m.data[3], m.data[7], m.data[11]};
    vec3 tr2 = -1 * (rot_trans * tr);
    mat4 m2 = {};

    for(int i = 0; i < 3; ++i)
    {
        m2.data[i*4 + 0] = rot_trans.data[i*3 + 0];
        m2.data[i*4 + 1] = rot_trans.data[i*3 + 1];
        m2.data[i*4 + 2] = rot_trans.data[i*3 + 2];
    }
    m2.data[3] = tr2.x;
    m2.data[7] = tr2.y;
    m2.data[11] = tr2.z;
    m2.data[15] = 1;
    return m2;
}

// transforms from world coordinates to camera coordinates
// note: this function could be implemented using invert_coord_change()
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
    return m * translate(-pos); // first translate then rotate
}

// with yaw = 0 and pitch = 0 camera points in the negative z direction (in a world space)
inline
mat4 lookat(vec3 pos, float yaw, float pitch)
{
    return rotate_y4(-yaw) * rotate_x4(-pitch) * translate(-pos);
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

inline
mat4 perspective(float fov_hori, float aspect, float near, float far)
{
    float right = near * tanf(fov_hori / 2);
    float top = right / aspect;
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

// mat = translate * rotate * scale
inline
void decompose(mat4 mat, vec3& pos, mat3& rot, vec3& scale)
{
    pos.x = mat.data[3];
    pos.y = mat.data[7];
    pos.z = mat.data[11];

    for(int col = 0; col < 3; ++col)
    {
        float v = 0;

        for(int row = 0; row < 3; ++row)
            v += mat.data[4*row + col] * mat.data[4*row + col];

        scale[col] = sqrtf(v);
    }

    for(int col = 0; col < 3; ++col)
    {
        for(int row = 0; row < 3; ++row)
            rot.data[3*row + col] = mat.data[4*row + col] / scale[col];
    }

    // fix the handness

    vec3 x1 = {rot.data[0], rot.data[3], rot.data[6]};
    vec3 y1 = {rot.data[1], rot.data[4], rot.data[7]};
    vec3 z1 = {rot.data[2], rot.data[5], rot.data[8]};
    vec3 z2 = cross(x1, y1);

    if(dot(z2, z1) < 0)
    {
        rot.data[2] = z2.x;
        rot.data[5] = z2.y;
        rot.data[8] = z2.z;
        scale.z *= -1;
    }
}
