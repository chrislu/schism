
#include <algorithm>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline vec<scal_type, 4>::vec()
{
}

template<typename scal_type>
inline vec<scal_type, 4>::vec(const vec<scal_type, 4>& v)
  : x(v.x), y(v.y), z(v.z), w(v.w)
{
}

template<typename scal_type>
inline vec<scal_type, 4>::vec(const scal_type s)
  : x(s), y(s), z(s), w(s) 
{
}

template<typename scal_type>
inline vec<scal_type, 4>::vec(const scal_type _x,
                              const scal_type _y,
                              const scal_type _z,
                              const scal_type _w)
  : x(_x), y(_y), z(_z), w(_w)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 4>::vec(const vec<rhs_scal_t, 4>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y)),
    z(static_cast<scal_type>(v.z)),
    w(static_cast<scal_type>(v.w))
{
}

// swap
template<typename scal_type>
inline void vec<scal_type, 4>::swap(vec<scal_type, 4>& rhs)
{
    std::swap_ranges(data_array, data_array + 4, rhs.data_array);
}

// assign
template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator=(const vec<scal_type, 4>& rhs)
{
    // performance wise very bad!
    //vec<scal_type, 4> tmp(rhs);
    //swap(tmp);

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator=(const vec<rhs_scal_t, 4>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;

    return (*this);
}

// unary operators
template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator+=(const scal_type s)
{
    x += s;
    y += s;
    z += s;
    w += s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator+=(const vec<scal_type, 4>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;
    z -= s;
    w -= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator-=(const vec<scal_type, 4>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;
    z *= s;
    w *= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator*=(const vec<scal_type, 4>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;
    z /= s;
    w /= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator/=(const vec<scal_type, 4>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4> vec<scal_type, 4>::operator++(int)
{
    vec<scal_type, 4> tmp(*this);

    *this += scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator++()
{
    return (*this += scal_type(1));
}

template<typename scal_type>
inline vec<scal_type, 4> vec<scal_type, 4>::operator--(int)
{
    vec<scal_type, 4> tmp(*this);

    *this -= scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator--()
{
    return (*this -= scal_type(1));
}

template<typename scal_type>
inline const vec<scal_type, 4> operator-(const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(rhs) *= scal_type(-1));
}

// binary operators
template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator+(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs) += rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 4> operator+(const scal_type_l        lhs,
//                                         const vec<scal_type, 4>& rhs)
//{
//    return (vec<scal_type, 4>(lhs) += rhs);
//
//    //return (vec<scal_type, 4>(lhs + rhs.x,
//    //                          lhs + rhs.y,
//    //                          lhs + rhs.z,
//    //                          lhs + rhs.w));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator+(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs) += rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator-(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs) -= rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 4> operator-(const scal_type_l        lhs,
//                                         const vec<scal_type, 4>& rhs)
//{
//    return (vec<scal_type, 4>(lhs - rhs.x,
//                              lhs - rhs.y,
//                              lhs - rhs.z,
//                              lhs - rhs.w));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator-(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs) -= rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator*(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs) *= rhs);
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 4> operator*(const scal_type_l        lhs,
                                         const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(rhs) *= lhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator*(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs) *= rhs);
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator/(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs) /= rhs);
}

//template<typename scal_type, typename scal_type_l>
//inline const vec<scal_type, 4> operator/(const scal_type_l        lhs,
//                                         const vec<scal_type, 4>& rhs)
//{
//    return (vec<scal_type, 4>(lhs / rhs.x,
//                              lhs / rhs.y,
//                              lhs / rhs.z,
//                              lhs / rhs.w));
//}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator/(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs) /= rhs);
}

template<typename scal_type>
inline scal_type dot(const vec<scal_type, 4>& lhs, const vec<scal_type, 4>& rhs)
{
    return (  lhs.x * rhs.x
            + lhs.y * rhs.y
            + lhs.z * rhs.z
            + lhs.w * rhs.w);
}

template<typename scal_type>
inline const vec<scal_type, 3> cross(const vec<scal_type, 4>& lhs, const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 3>(lhs.y * rhs.z - lhs.z * rhs.y,
                              lhs.z * rhs.x - lhs.x * rhs.z,
                              lhs.x * rhs.y - lhs.y * rhs.x));
}



    template<typename scm_scalar, unsigned dim>
    const vec<scm_scalar, dim> clamp(const vec<scm_scalar, dim>& val, const vec<scm_scalar, dim>& min, const vec<scm_scalar, dim>& max)
    {
        vec<scm_scalar, dim> tmp_ret;

        for (unsigned i = 0; i < dim; i++) {
            tmp_ret[i] = ((val.vec_array[i] > max.vec_array[i]) ? max.vec_array[i] : (val.vec_array[i] < min.vec_array[i]) ? min.vec_array[i] : val.vec_array[i]);
        }

        return (tmp_ret);
    }

    template<typename scm_scalar, unsigned dim>
    const vec<scm_scalar, dim> pow(const vec<scm_scalar, dim>& val, scm_scalar exponent)
    {
        vec<scm_scalar, dim> tmp_ret;

        for (unsigned i = 0; i < dim; i++) {
            tmp_ret[i] = math::pow(val.vec_array[i], exponent);
        }

        return (tmp_ret);
    }


} // namespace math
} // namespace scm
