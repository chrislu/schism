
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

    enum component { _x = 0,
                     _y = 1};

public:
    // ctors
    vec();
    vec(const vec<scal_type, 2>& v);
    explicit vec(const scal_type s);
    explicit vec(const scal_type s,
                 const scal_type t);

    template<typename rhs_scal_t> explicit vec(const vec<rhs_scal_t, 2>& v);

    // swap
    void swap(vec<scal_type, 2>& rhs);

    // assign
    vec<scal_type, 2>&                               operator=(const vec<scal_type, 2>& rhs);
    template<typename rhs_scal_t> vec<scal_type, 2>& operator=(const vec<rhs_scal_t, 2>& rhs);

    // data access
    inline scal_type*const         operator&()          { return (data_array); }
    inline const scal_type*const   operator&() const    { return (data_array); }

    // index
    inline scal_type& operator[](const component i)         { return data_array[i]; };
    inline scal_type  operator[](const component i) const   { return data_array[i]; };

    // unary operators
    vec<scal_type, 2>& operator+=(const scal_type          s);
    vec<scal_type, 2>& operator+=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>& operator-=(const scal_type          s);
    vec<scal_type, 2>& operator-=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>& operator*=(const scal_type          s);
    vec<scal_type, 2>& operator*=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>& operator/=(const scal_type          s);
    vec<scal_type, 2>& operator/=(const vec<scal_type, 2>& v);
    vec<scal_type, 2>  operator++(int);
    vec<scal_type, 2>& operator++();
    vec<scal_type, 2>  operator--(int);
    vec<scal_type, 2>& operator--();

    // data definition
    union {
        struct {scal_type x, y;};
        struct {scal_type r, g;};
        struct {scal_type s, t;};
        scal_type data_array[2];
    };

}; // class vec<scal_type, 2>

} // namespace math
} // namespace scm

#include "vec2.inl"

#endif // MATH_VEC2_H_INCLUDED
