
#include <algorithm>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline vec<scal_type, 2>::vec()
{
}

template<typename scal_type>
inline vec<scal_type, 2>::vec(const vec<scal_type, 2>& v)
  : x(v.x), y(v.y)
{
}

template<typename scal_type>
inline vec<scal_type, 2>::vec(const scal_type s)
  : x(s), y(s) 
{
}

template<typename scal_type>
inline vec<scal_type, 2>::vec(const scal_type _x,
                              const scal_type _y)
  : x(_x), y(_y)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 2>::vec(const vec<rhs_scal_t, 2>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y))
{
}

// swap
template<typename scal_type>
inline void vec<scal_type, 2>::swap(vec<scal_type, 2>& rhs)
{
    std::swap_ranges(data_array, data_array + 2, rhs.data_array);
}

// assign
template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator=(const vec<scal_type, 2>& rhs)
{
    // performance wise very bad!
    //vec<scal_type, 2> tmp(rhs);
    //swap(tmp);

    x = rhs.x;
    y = rhs.y;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator=(const vec<rhs_scal_t, 2>& rhs)
{
    x = rhs.x;
    y = rhs.y;

    return (*this);
}

// unary operators
template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator+=(const scal_type s)
{
    x += s;
    y += s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator+=(const vec<scal_type, 2>& v)
{
    x += v.x;
    y += v.y;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator-=(const vec<scal_type, 2>& v)
{
    x -= v.x;
    y -= v.y;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator*=(const vec<scal_type, 2>& v)
{
    x *= v.x;
    y *= v.y;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator/=(const vec<scal_type, 2>& v)
{
    x /= v.x;
    y /= v.y;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 2> vec<scal_type, 2>::operator++(int)
{
    vec<scal_type, 2> tmp(*this);

    *this += scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator++()
{
    return (*this += scal_type(1));
}

template<typename scal_type>
inline vec<scal_type, 2> vec<scal_type, 2>::operator--(int)
{
    vec<scal_type, 2> tmp(*this);

    *this -= scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 2>& vec<scal_type, 2>::operator--()
{
    return (*this -= scal_type(1));
}

template<typename scal_type>
inline const vec<scal_type, 2> operator-(const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(rhs) *= scal_type(-1));
}

// binary operators
template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator+(const vec<scal_type,   2>& lhs,
                                         const vec<scal_type_r, 2>& rhs)
{
    return (vec<scal_type, 2>(lhs) += rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 2> operator+(const scal_type_l        lhs,
//                                         const vec<scal_type, 2>& rhs)
//{
//    return (vec<scal_type, 2>(lhs) += rhs);
//
//    //return (vec<scal_type, 2>(lhs + rhs.x,
//    //                          lhs + rhs.y));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator+(const vec<scal_type, 2>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 2>(lhs) += rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator-(const vec<scal_type,   2>& lhs,
                                         const vec<scal_type_r, 2>& rhs)
{
    return (vec<scal_type, 2>(lhs) -= rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 2> operator-(const scal_type_l        lhs,
//                                         const vec<scal_type, 2>& rhs)
//{
//    return (vec<scal_type, 2>(lhs - rhs.x,
//                              lhs - rhs.y));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator-(const vec<scal_type, 2>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 2>(lhs) -= rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator*(const vec<scal_type,   2>& lhs,
                                         const vec<scal_type_r, 2>& rhs)
{
    return (vec<scal_type, 2>(lhs) *= rhs);
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 2> operator*(const scal_type_l        lhs,
                                         const vec<scal_type, 2>& rhs)
{
    return (vec<scal_type, 2>(rhs) *= lhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator*(const vec<scal_type, 2>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 2>(lhs) *= rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator/(const vec<scal_type,   2>& lhs,
                                         const vec<scal_type_r, 2>& rhs)
{
    return (vec<scal_type, 2>(lhs) /= rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 2> operator/(const scal_type_l        lhs,
//                                         const vec<scal_type, 2>& rhs)
//{
//    return (vec<scal_type, 2>(lhs / rhs.x,
//                              lhs / rhs.y));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 2> operator/(const vec<scal_type, 2>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 2>(lhs) /= rhs);
}

} // namespace math
} // namespace scm
