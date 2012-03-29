
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_MAT3_H_INCLUDED
#define MATH_MAT3_H_INCLUDED

#include "mat.h"

#include <scm/core/math/vec_fwd.h>

namespace scm {
namespace math {

template<typename scal_type>
class mat<scal_type, 3, 3>
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    mat();
    mat(const mat<scal_type, 3, 3>& m);
    //mat(const scal_type a[9]);

    explicit mat(const scal_type a00, const scal_type a01, const scal_type a02,
                 const scal_type a03, const scal_type a04, const scal_type a05,
                 const scal_type a06, const scal_type a07, const scal_type a08);

    explicit mat(const vec<scal_type, 3>& c00,
                 const vec<scal_type, 3>& c01,
                 const vec<scal_type, 3>& c02);

    template<typename rhs_scal_t>
    explicit mat(const mat<rhs_scal_t, 3, 3>& m);

    // constants
    static const mat<scal_type, 3, 3>&  zero();
    static const mat<scal_type, 3, 3>&  identity();

    // dtor
    //~mat();

    // swap
    void swap(mat<scal_type, 3, 3>& rhs);

    // assign
    mat<scal_type, 3, 3>&           operator=(const mat<scal_type, 3, 3>& rhs);
    template<typename rhs_scal_t>
    mat<scal_type, 3, 3>&           operator=(const mat<rhs_scal_t, 3, 3>& rhs);

    // data access
    //inline scal_type*const          operator&()         { return (data_array); }
    //inline const scal_type*const    operator&() const   { return (data_array); }

    // index
    inline scal_type&               operator[](const int i);
    inline scal_type                operator[](const int i) const;

    inline vec<scal_type, 3>        column(const int i) const;
    inline vec<scal_type, 3>        row(const int i) const;

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
        };
        scal_type  data_array[9];
    };

}; // class mat<scal_type, 3, 3>


} // namespace math
} // namespace scm

#include "mat3.inl"

#endif // MATH_MAT3_H_INCLUDED
