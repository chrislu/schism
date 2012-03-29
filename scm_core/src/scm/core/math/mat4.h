
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_MAT4_H_INCLUDED
#define MATH_MAT4_H_INCLUDED

#include "mat.h"

#include <scm/core/math/vec_fwd.h>

namespace scm {
namespace math {

template<typename scal_type>
class mat<scal_type, 4, 4>
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    mat();
    mat(const mat<scal_type, 4, 4>& m);
    //mat(const scal_type a[16]);

    explicit mat(const scal_type a00, const scal_type a01, const scal_type a02, const scal_type a03,
                 const scal_type a04, const scal_type a05, const scal_type a06, const scal_type a07,
                 const scal_type a08, const scal_type a09, const scal_type a10, const scal_type a11,
                 const scal_type a12, const scal_type a13, const scal_type a14, const scal_type a15);

    explicit mat(const vec<scal_type, 4>& c00,
                 const vec<scal_type, 4>& c01,
                 const vec<scal_type, 4>& c02,
                 const vec<scal_type, 4>& c03);

    template<typename rhs_scal_t>
    explicit mat(const mat<rhs_scal_t, 4, 4>& m);

    // constants
    static const mat<scal_type, 4, 4>&  zero();
    static const mat<scal_type, 4, 4>&  identity();

    // dtor
    //~mat();

    // swap
    void swap(mat<scal_type, 4, 4>& rhs);

    // assign
    mat<scal_type, 4, 4>&           operator=(const mat<scal_type, 4, 4>& rhs);
    template<typename rhs_scal_t>
    mat<scal_type, 4, 4>&           operator=(const mat<rhs_scal_t, 4, 4>& rhs);

    // data access
    //inline scal_type*const          operator&()         { return (data_array); }
    //inline const scal_type*const    operator&() const   { return (data_array); }

    // index
    inline scal_type&               operator[](const int i);
    inline scal_type                operator[](const int i) const;

    inline vec<scal_type, 4>        column(const int i) const;
    inline vec<scal_type, 4>        row(const int i) const;

    // data definition
    union
    {
        struct
        {
            scal_type m00;
            scal_type m01;
            scal_type m02;
            scal_type m03;
            scal_type m04;
            scal_type m05;
            scal_type m06;
            scal_type m07;
            scal_type m08;
            scal_type m09;
            scal_type m10;
            scal_type m11;
            scal_type m12;
            scal_type m13;
            scal_type m14;
            scal_type m15;
        };
        scal_type  data_array[16];
    };

}; // class mat<scal_type, 4, 4>


} // namespace math
} // namespace scm

#include "mat4.inl"

#endif // MATH_MAT4_H_INCLUDED
