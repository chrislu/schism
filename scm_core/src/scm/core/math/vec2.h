
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_VEC2_H_INCLUDED
#define MATH_VEC2_H_INCLUDED

#include "vec.h"

namespace scm {
namespace math {

template<typename scal_type>
class vec<scal_type, 2>
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    vec();
    vec(const vec<scal_type, 2>& v);
    //vec(const scal_type a[2]);

    vec(const vec<scal_type, 3>& v);
    vec(const vec<scal_type, 4>& v);

    explicit vec(const scal_type s);
    explicit vec(const scal_type s,
                 const scal_type t);

    template<typename rhs_scal_t> explicit vec(const vec<rhs_scal_t, 2>& v);

    // dtor
    //~vec();

    // constants
    static const vec<scal_type, 2>&  zero();
    static const vec<scal_type, 2>&  one();

    // swap
    void swap(vec<scal_type, 2>& rhs);

    // assign
    vec<scal_type, 2>&              operator=(const vec<scal_type, 2>& rhs);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator=(const vec<rhs_scal_t, 2>& rhs);

    // data access
    //inline scal_type*const          operator&()          { return (data_array); }
    //inline const scal_type*const    operator&() const    { return (data_array); }

    // index
    inline scal_type&               operator[](const int i)         { return data_array[i]; };
    inline scal_type                operator[](const int i) const   { return data_array[i]; };

    // unary operators
    vec<scal_type, 2>&              operator+=(const scal_type          s);
    vec<scal_type, 2>&              operator+=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>&              operator-=(const scal_type          s);
    vec<scal_type, 2>&              operator-=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>&              operator*=(const scal_type          s);
    vec<scal_type, 2>&              operator*=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>&              operator/=(const scal_type          s);
    vec<scal_type, 2>&              operator/=(const vec<scal_type, 2>& v);
    bool                            operator==(const vec<scal_type, 2>& v) const;
    bool                            operator!=(const vec<scal_type, 2>& v) const;

    // unary operators
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator+=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator+=(const vec<rhs_scal_t, 2>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator-=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator-=(const vec<rhs_scal_t, 2>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator*=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator*=(const vec<rhs_scal_t, 2>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator/=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 2>&              operator/=(const vec<rhs_scal_t, 2>& v);

    // data definition
    union {
        struct {scal_type x, y;};
        struct {scal_type r, g;};
        struct {scal_type s, t;};
        scal_type data_array[2];
    };

}; // class vec<scal_type, 2>

// common functions
template<typename scal_type> scal_type                      dot(const vec<scal_type, 2>& lhs, const vec<scal_type, 2>& rhs);
template<typename scal_type> const vec<scal_type, 2>        cross(const vec<scal_type, 2>& lhs, const vec<scal_type, 2>& rhs);
template<typename scal_type> const vec<scal_type, 2>        clamp(const vec<scal_type, 2>& val, const vec<scal_type, 2>& min, const vec<scal_type, 2>& max);
template<typename scal_type> const vec<scal_type, 2>        pow(const vec<scal_type, 2>& val, const scal_type exp);
template<typename scal_type> const vec<scal_type, 2>        min(const vec<scal_type, 2>& a, const vec<scal_type, 2>& b);
template<typename scal_type> const vec<scal_type, 2>        max(const vec<scal_type, 2>& a, const vec<scal_type, 2>& b);
template<typename scal_type> const vec<scal_type, 2>        floor(const vec<scal_type, 2>& rhs);
template<typename scal_type> const vec<scal_type, 2>        ceil(const vec<scal_type, 2>& rhs);
template<typename scal_type> const vec<scal_type, 2>        fract(const vec<scal_type, 2>& rhs);

} // namespace math
} // namespace scm

#include "vec2.inl"

#endif // MATH_VEC2_H_INCLUDED
