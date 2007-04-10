
#ifndef SCM_MATH_MAT_H_INCLUDED
#define SCM_MATH_MAT_H_INCLUDED

#include "math_vec.h"

namespace math
{
    // matrices in column-major layout

    template<typename scm_scalar, unsigned row_dim, unsigned col_dim>
    class mat
    {
    public:
        typedef scm_scalar          scal_type;
    };

    // savety for some recursive library functions
    template<typename scm_scalar>
    class mat<scm_scalar, 1, 1>
    {
    public:
        //// data definition
        union
        {
            struct
            {
                scm_scalar m00;
            };
            scm_scalar  mat_array[1];
        };
    };

    template<typename scm_scalar>
    class mat<scm_scalar, 2, 2>
    {
    public:
        mat() {}
        mat(const mat<scm_scalar, 2, 2>& m) : m00(m.m00), m01(m.m01), m02(m.m02), m03(m.m03) {}
        explicit mat(const scm_scalar s) : m00(s), m01(s), m02(s), m03(s) {}
        explicit mat(const scm_scalar a00,
                     const scm_scalar a01,
                     const scm_scalar a02,
                     const scm_scalar a03) : m00(a00), m01(a01), m02(a02), m03(a03) {}
        explicit mat(const vec<scm_scalar, 2>& c00,
                     const vec<scm_scalar, 2>& c01) : m00(c00.x), m01(c00.y), m02(c01.x), m03(c01.y) {}

        //vec<scm_scalar, 2>& operator[](const unsigned i) { assert(i < 2); return (columns[i]); }
        //const vec<scm_scalar, 2>& operator[](const unsigned i) const { assert(i < 2); return (columns[i]); }

        //// data definition
        union
        {
            struct
            {
                scm_scalar m00;
                scm_scalar m01;
                scm_scalar m02;
                scm_scalar m03;
            };
            /*struct
            {
                vec<scm_scalar, 2>    columns[2];
            };*/
            scm_scalar  mat_array[2 * 2];
        };
    protected:
    private:
    }; // class mat2x2

    template<typename scm_scalar>
    class mat<scm_scalar, 3, 3>
    {
    public:
        mat() {}
        mat(const mat<scm_scalar, 3, 3>& m) : m00(m.m00), m01(m.m01), m02(m.m02),
                                              m03(m.m03), m04(m.m04), m05(m.m05),
                                              m06(m.m06), m07(m.m07), m08(m.m08) {}
        explicit mat(const scm_scalar s)  : m00(s), m01(s), m02(s),
                                            m03(s), m04(s), m05(s),
                                            m06(s), m07(s), m08(s) {}
        explicit mat(const scm_scalar a00,
                     const scm_scalar a01,
                     const scm_scalar a02,
                     const scm_scalar a03,
                     const scm_scalar a04,
                     const scm_scalar a05,
                     const scm_scalar a06,
                     const scm_scalar a07,
                     const scm_scalar a08)  : m00(a00), m01(a01), m02(a02),
                                              m03(a03), m04(a04), m05(a05),
                                              m06(a06), m07(a07), m08(a08) {}
        explicit mat(const vec<scm_scalar, 3>& c00,
                     const vec<scm_scalar, 3>& c01,
                     const vec<scm_scalar, 3>& c02)  : m00(c00.x), m01(c00.y), m02(c00.z),
                                                       m03(c01.x), m04(c01.y), m05(c01.z),
                                                       m06(c02.x), m07(c02.y), m08(c02.z) {}

        //vec<scm_scalar, 3>& operator[](const unsigned i) { assert(i < 3); return (columns[i]); }
        //const vec<scm_scalar, 3>& operator[](const unsigned i) const { assert(i < 3); return (columns[i]); }

        // data definition
        union
        {
            struct
            {
                scm_scalar m00;
                scm_scalar m01;
                scm_scalar m02;
                scm_scalar m03;
                scm_scalar m04;
                scm_scalar m05;
                scm_scalar m06;
                scm_scalar m07;
                scm_scalar m08;
            };
            /*struct
            {
                vec<scm_scalar, 3>    columns[3];
            };*/
            scm_scalar  mat_array[3 * 3];
        };
    protected:
    private:
    }; // class mat3x3


    template<typename scm_scalar>
    class mat<scm_scalar, 4, 4>
    {
    public:
        mat() {}
        mat(const mat<scm_scalar, 4, 4>& m) : m00(m.m00), m01(m.m01), m02(m.m02), m03(m.m03),
                                              m04(m.m04), m05(m.m05), m06(m.m06), m07(m.m07),
                                              m08(m.m08), m09(m.m09), m10(m.m10), m11(m.m11),
                                              m12(m.m12), m13(m.m13), m14(m.m14), m15(m.m15) {}
        explicit mat(const scm_scalar  s) : m00(s), m01(s), m02(s), m03(s),
                                            m04(s), m05(s), m06(s), m07(s),
                                            m08(s), m09(s), m10(s), m11(s),
                                            m12(s), m13(s), m14(s), m15(s) {}
        explicit mat(const scm_scalar a00,
                     const scm_scalar a01,
                     const scm_scalar a02,
                     const scm_scalar a03,
                     const scm_scalar a04,
                     const scm_scalar a05,
                     const scm_scalar a06,
                     const scm_scalar a07,
                     const scm_scalar a08,
                     const scm_scalar a09,
                     const scm_scalar a10,
                     const scm_scalar a11,
                     const scm_scalar a12,
                     const scm_scalar a13,
                     const scm_scalar a14,
                     const scm_scalar a15)  : m00(a00), m01(a01), m02(a02), m03(a03),
                                              m04(a04), m05(a05), m06(a06), m07(a07),
                                              m08(a08), m09(a09), m10(a10), m11(a11),
                                              m12(a12), m13(a13), m14(a14), m15(a15) {}
        explicit mat(const vec<scm_scalar, 4>& c00,
                     const vec<scm_scalar, 4>& c01,
                     const vec<scm_scalar, 4>& c02,
                     const vec<scm_scalar, 4>& c03) : m00(c00.x), m01(c00.y), m02(c00.z), m03(c00.w),
                                                      m04(c01.x), m05(c01.y), m06(c01.z), m07(c01.w),
                                                      m08(c02.x), m09(c02.y), m10(c02.z), m11(c02.w),
                                                      m12(c03.x), m13(c03.y), m14(c03.z), m15(c03.w)  {}

        //vec<scm_scalar, 4>& operator[](const unsigned i) { assert(i < 4); return (columns[i]); }
        //const vec<scm_scalar, 4>& operator[](const unsigned i) const { assert(i < 4); return (columns[i]); }

        // data definition
        union
        {
            struct
            {
                scm_scalar m00;
                scm_scalar m01;
                scm_scalar m02;
                scm_scalar m03;
                scm_scalar m04;
                scm_scalar m05;
                scm_scalar m06;
                scm_scalar m07;
                scm_scalar m08;
                scm_scalar m09;
                scm_scalar m10;
                scm_scalar m11;
                scm_scalar m12;
                scm_scalar m13;
                scm_scalar m14;
                scm_scalar m15;
            };
            /*struct
            {
                vec<scm_scalar, 4>    columns[4];
            };*/
            scm_scalar  mat_array[4 * 4];
        };
    protected:
    private:
    }; // class mat4x4

} // namespace math

#include "math_mat.inl"

#endif // SCM_MATH_MAT_H_INCLUDED



