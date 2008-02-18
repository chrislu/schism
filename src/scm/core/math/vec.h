
#ifndef VEC_H_INLCUDED
#define VEC_H_INLCUDED

#include <cassert>

#include <boost/static_assert.hpp>

namespace scm {
namespace math {

template<typename scm_scalar, unsigned dim>
class vec
{
}; // class vec<scm_scalar, dim>

// vec4
//template<typename scm_scalar>
//class vec<scm_scalar, 4>
class vec4f
{
public:
    //typedef float scm_scalar;

public:
    vec4f() {}
    //explicit vec4f(const float s) : x(s), y(s), z(s), w(s) {}
    explicit vec4f(const float s, const float t, const float u, const float v) : x(s), y(t), z(u), w(v) {}

    template<class T> void evaluate(const T& e) {x = e[0]; y = e[1]; z = e[2]; w = e[3];}
    template<class T> vec4f(const T&e) {evaluate(e);}
    template<class T> vec4f& operator=(const T& e) {evaluate(e); return (*this);}

    //template<typename scal_type> vec(const vec<scal_type, 4>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    //template<typename scal_type> vec(const vec<scal_type, 3>& v) : x(v.x), y(v.y), z(v.z), w(scal_type(1)) {}
    //template<typename scal_type> vec(const vec<scal_type, 3>& v, const scal_type s) : x(v.x), y(v.y), z(v.z), w(s) {}

    //template<typename scal_type> inline vec<scm_scalar, 4>& operator=(const vec<scal_type, 4>& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return (*this); }
    //inline vec4f& operator=(const vec4f& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return (*this); }

    inline float& operator[](const int i)            { return vec_array[i]; };
    inline float  operator[](const int i) const      { return vec_array[i]; };

    // data definition
    union {
        struct {
            float x;
            float y;
            float z;
            float w;
        };
        float  vec_array[4];
    };

}; // class vec<scm_scalar, 4>

template<typename scm_scalar>
class vec<scm_scalar, 4>
{
public:
    //typedef float scm_scalar;

public:
    vec() {}
    explicit vec(const scm_scalar s) : x(s), y(s), z(s), w(s) {}
    explicit vec(const scm_scalar s, const scm_scalar t, const scm_scalar u, const scm_scalar v) : x(s), y(t), z(u), w(v) {}

    //template<typename scal_type> vec(const vec<scal_type, 4>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    //template<typename scal_type> vec(const vec<scal_type, 3>& v) : x(v.x), y(v.y), z(v.z), w(scal_type(1)) {}
    //template<typename scal_type> vec(const vec<scal_type, 3>& v, const scal_type s) : x(v.x), y(v.y), z(v.z), w(s) {}

    template<typename scal_type> inline vec<scm_scalar, 4>& operator=(const vec<scal_type, 4>& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return (*this); }
    //inline vec4f& operator=(const vec4f& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return (*this); }

    inline scm_scalar& operator[](const int i)            { return vec_array[i]; };
    inline scm_scalar  operator[](const int i) const      { return vec_array[i]; };

    // data definition
    union {
        struct {
            float x;
            float y;
            float z;
            float w;
        };
        float  vec_array[4];
    };

}; // class vec<scm_scalar, 4>
// temporary
typedef vec<float, 4>       vec4f_c;
//typedef vec<double, 4>      vec4d;


} // namespace math
} // namespace scm

#include "vec.inl"

#endif // VEC_H_INLCUDED
