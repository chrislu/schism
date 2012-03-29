
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_MAT2_H_INCLUDED
#define MATH_MAT2_H_INCLUDED

#include "mat.h"

#include <scm/core/math/vec_fwd.h>

namespace scm {
namespace math {

template<typename scal_type>
class mat<scal_type, 2, 2>
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    mat();
    mat(const mat<scal_type, 2, 2>& m);
    //mat(const scal_type a[4]);

    explicit mat(const scal_type a00, const scal_type a01,
                 const scal_type a02, const scal_type a03);

    explicit mat(const vec<scal_type, 2>& c00,
                 const vec<scal_type, 2>& c01);

    template<typename rhs_scal_t>
    explicit mat(const mat<rhs_scal_t, 2, 2>& m);

    // constants
    static const mat<scal_type, 2, 2>&  zero();
    static const mat<scal_type, 2, 2>&  identity();

    // dtor
    //~mat();

    // swap
    void swap(mat<scal_type, 2, 2>& rhs);

    // assign
    mat<scal_type, 2, 2>&           operator=(const mat<scal_type, 2, 2>& rhs);
    template<typename rhs_scal_t>
    mat<scal_type, 2, 2>&           operator=(const mat<rhs_scal_t, 2, 2>& rhs);

    // data access
    //inline scal_type*const          operator&()         { return (data_array); }
    //inline const scal_type*const    operator&() const   { return (data_array); }

    // index
    inline scal_type&               operator[](const int i);
    inline scal_type                operator[](const int i) const;

    inline vec<scal_type, 2>        column(const int i) const;
    inline vec<scal_type, 2>        row(const int i) const;

    // data definition
    union
    {
        struct
        {
            scal_type m00;
            scal_type m01;
            scal_type m02;
            scal_type m03;
        };
        scal_type  data_array[4];
    };

}; // class mat<scal_type, 2, 2>

} // namespace math
} // namespace scm

#include "mat2.inl"

#endif // MATH_MAT2_H_INCLUDED
