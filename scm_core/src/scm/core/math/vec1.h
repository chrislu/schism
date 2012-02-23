
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef MATH_VEC1_H_INCLUDED
#define MATH_VEC1_H_INCLUDED

#include "vec.h"

namespace scm {
namespace math {

template<typename scal_type>
class vec<scal_type, 1>
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    vec();
    vec(const vec<scal_type, 1>& v);
    //vec(const scal_type a[2]);

    vec(const vec<scal_type, 2>& v);
    vec(const vec<scal_type, 3>& v);
    vec(const vec<scal_type, 4>& v);

    explicit vec(const scal_type s);

    template<typename rhs_scal_t> explicit vec(const vec<rhs_scal_t, 1>& v);

    // dtor
    //~vec();

    // constants
    static const vec<scal_type, 1>&  zero();
    static const vec<scal_type, 1>&  one();

    // swap
    void swap(vec<scal_type, 1>& rhs);

    // assign
    vec<scal_type, 1>&              operator=(const vec<scal_type, 1>& rhs);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator=(const vec<rhs_scal_t, 1>& rhs);

    // data access
    //inline scal_type*const          operator&()          { return (data_array); }
    //inline const scal_type*const    operator&() const    { return (data_array); }

    // index
    inline scal_type&               operator[](const int i)         { return data_array[i]; };
    inline scal_type                operator[](const int i) const   { return data_array[i]; };

    // unary operators
    vec<scal_type, 1>&              operator+=(const scal_type          s);
    vec<scal_type, 1>&              operator+=(const vec<scal_type, 1>& v);
    vec<scal_type, 1>&              operator-=(const scal_type          s);
    vec<scal_type, 1>&              operator-=(const vec<scal_type, 1>& v);
    vec<scal_type, 1>&              operator*=(const scal_type          s);
    vec<scal_type, 1>&              operator*=(const vec<scal_type, 1>& v);
    vec<scal_type, 1>&              operator/=(const scal_type          s);
    vec<scal_type, 1>&              operator/=(const vec<scal_type, 1>& v);
    bool                            operator==(const vec<scal_type, 1>& v) const;
    bool                            operator!=(const vec<scal_type, 1>& v) const;

    // unary operators
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator+=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator+=(const vec<rhs_scal_t, 1>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator-=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator-=(const vec<rhs_scal_t, 1>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator*=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator*=(const vec<rhs_scal_t, 1>& v);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator/=(const rhs_scal_t          s);
    template<typename rhs_scal_t>
    vec<scal_type, 1>&              operator/=(const vec<rhs_scal_t, 1>& v);

    // data definition
    union {
        struct {scal_type x;};
        struct {scal_type r;};
        struct {scal_type s;};
        scal_type data_array[1];
    };

}; // class vec<scal_type, 1>

// common functions
template<typename scal_type> scal_type                      dot(const vec<scal_type, 1>& lhs, const vec<scal_type, 1>& rhs);
template<typename scal_type> const vec<scal_type, 1>        clamp(const vec<scal_type, 1>& val, const vec<scal_type, 1>& min, const vec<scal_type, 1>& max);
template<typename scal_type> const vec<scal_type, 1>        pow(const vec<scal_type, 1>& val, const scal_type exp);
template<typename scal_type> const vec<scal_type, 1>        min(const vec<scal_type, 1>& a, const vec<scal_type, 1>& b);
template<typename scal_type> const vec<scal_type, 1>        max(const vec<scal_type, 1>& a, const vec<scal_type, 1>& b);
template<typename scal_type> const vec<scal_type, 1>        floor(const vec<scal_type, 1>& rhs);
template<typename scal_type> const vec<scal_type, 1>        ceil(const vec<scal_type, 1>& rhs);
template<typename scal_type> const vec<scal_type, 1>        fract(const vec<scal_type, 1>& rhs);

} // namespace math
} // namespace scm

#include "vec1.inl"

#endif // MATH_VEC1_H_INCLUDED
