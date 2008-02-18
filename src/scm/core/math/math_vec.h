
#ifndef SCM_MATH_VEC_H_INCLUDED
#define SCM_MATH_VEC_H_INCLUDED

#include <cassert>

#include <boost/static_assert.hpp>

namespace math
{
    template<typename scm_scalar, unsigned dim>
    class vec
    {
    public:
        vec() {}
        vec(const vec<scm_scalar, dim>& v) {
            for(unsigned i = 0; i < dim; i++) {
                vec_array[i] = v.vec_array[i];
            }
        }
        explicit vec(const scm_scalar s) {
            for(unsigned i = 0; i < dim; i++) {
                vec_array[i] = s;
            }
        }

        vec<scm_scalar, dim>& operator=(const vec<scm_scalar, dim>& rhs) { 
            for(unsigned i = 0; i < dim; i++) {
                vec_array[i] = rhs.vec_array[i];
            }

            return (*this);
        }

        scm_scalar& operator[](const unsigned i) { assert(i < dim); return (vec_array[i]); }
        const scm_scalar& operator[](const unsigned i) const { assert(i < dim); return (vec_array[i]); }

        //// do not instantiiate!
        //BOOST_STATIC_ASSERT(sizeof(scm_scalar) == 0);
        scm_scalar  vec_array[dim];
    };

#if 0 // convenience
    template<typename scm_scalar>
    class vec<scm_scalar, 1>
    {
    public:
        vec() {}
        vec(const vec<scm_scalar, 1>& v) : x(v.x) {}
        explicit vec(const scm_scalar s) : x(s) {}
        
        vec<scm_scalar, 1>& operator=(const vec<scm_scalar, 1>& rhs) { x = rhs.x; return (*this); }

        scm_scalar& operator[](const unsigned i) { assert(i < 1); return (vec_array[i]); }
        const scm_scalar& operator[](const unsigned i) const { assert(i < 1); return (vec_array[i]); }

        // data definition
        union
        {
            struct
            {
                scm_scalar x;
            };
            scm_scalar  vec_array[1];
        };
    protected:
    private:
    }; // class vec1
#endif

    template<typename scm_scalar>
    class vec<scm_scalar, 2>
    {
    public:
        typedef scm_scalar          component_type;

    public:
        vec() {}
        vec(const vec<scm_scalar, 2>& v) : x(v.x), y(v.y) {}
        explicit vec(const scm_scalar s) : x(s), y(s) {}
        explicit vec(const scm_scalar s, const scm_scalar t) : x(s), y(t) {}

        vec<scm_scalar, 2>& operator=(const vec<scm_scalar, 2>& rhs) { x = rhs.x; y = rhs.y; return (*this); }

        scm_scalar& operator[](const unsigned i) { assert(i < 2); return (vec_array[i]); }
        const scm_scalar& operator[](const unsigned i) const { assert(i < 2); return (vec_array[i]); }

        // data definition
        union
        {
            struct
            {
                scm_scalar x;
                scm_scalar y;
            };
            scm_scalar  vec_array[2];
        };
    protected:
    private:
    }; // class vec2
    
    template<typename scm_scalar>
    class vec<scm_scalar, 3>
    {
    public:
        typedef scm_scalar          component_type;

    public:
        vec() {}
        vec(const vec<scm_scalar, 3>& v) : x(v.x), y(v.y), z(v.z) {}
        explicit vec(const scm_scalar s) : x(s), y(s), z(s) {}
        explicit vec(const scm_scalar s, const scm_scalar t, const scm_scalar u) : x(s), y(t), z(u) {}

        vec<scm_scalar, 3>& operator=(const vec<scm_scalar, 3>& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; return (*this); }

        scm_scalar& operator[](const unsigned i) { assert(i < 3); return (vec_array[i]); }
        const scm_scalar& operator[](const unsigned i) const { assert(i < 3); return (vec_array[i]); }

        // data definition
        union
        {
            struct
            {
                scm_scalar x;
                scm_scalar y;
                scm_scalar z;
            };
            scm_scalar  vec_array[3];
        };
    protected:
    private:
    }; // class vec3

    template<typename scm_scalar>
    class vec<scm_scalar, 4>
    {
    public:
        typedef scm_scalar          component_type;

    public:
        vec() {}
        vec(const vec<scm_scalar, 4>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
        explicit vec(const scm_scalar s) : x(s), y(s), z(s), w(s) {}
        explicit vec(const scm_scalar s, const scm_scalar t, const scm_scalar u, const scm_scalar v) : x(s), y(t), z(u), w(v) {}
        explicit vec(const vec<scm_scalar, 3>& v) : x(v.x), y(v.y), z(v.z), w(scm_scalar(1)) {}
        explicit vec(const vec<scm_scalar, 3>& v, const scm_scalar s) : x(v.x), y(v.y), z(v.z), w(s) {}

        vec<scm_scalar, 4>& operator=(const vec<scm_scalar, 4>& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return (*this); }

        scm_scalar& operator[](const unsigned i) { assert(i < 4); return (vec_array[i]); }
        const scm_scalar& operator[](const unsigned i) const { assert(i < 4); return (vec_array[i]); }

        // data definition
        union
        {
            struct
            {
                union {scm_scalar x, r, s;};
                union {scm_scalar y, g, t;};
                union {scm_scalar z, b, p;};
                union {scm_scalar w, a, q;};
            //    scm_scalar x;
            //    scm_scalar y;
            //    scm_scalar z;
            //    scm_scalar w;
            };
            scm_scalar  vec_array[4];
        };
    protected:
    private:
    }; // class vec4


} // math

#include "math_vec.inl"

#endif // SCM_MATH_VEC_H_INCLUDED

