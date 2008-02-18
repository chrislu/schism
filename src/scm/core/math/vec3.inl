
namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline vec<scal_type, 3>::vec()
{
}

template<typename scal_type>
inline vec<scal_type, 3>::vec(const vec<scal_type, 3>& v)
  : x(v.x), y(v.y), z(v.z)
{
}

template<typename scal_type>
inline vec<scal_type, 3>::vec(const scal_type s)
  : x(s), y(s), z(s) 
{
}

template<typename scal_type>
inline vec<scal_type, 3>::vec(const scal_type _x,
                              const scal_type _y,
                              const scal_type _z)
  : x(_x), y(_y), z(_z)
{
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 3>::vec(const vec<rhs_scal_t, 3>& v)
  : x(static_cast<scal_type>(v.x)),
    y(static_cast<scal_type>(v.y)),
    z(static_cast<scal_type>(v.z))
{
}

// assign
template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator=(const vec<scal_type, 3>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator=(const vec<rhs_scal_t, 3>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;

    return (*this);
}

// unary operators
template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator+=(const scal_type s)
{
    x += s;
    y += s;
    z += s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator+=(const vec<scal_type, 3>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator-=(const scal_type s)
{
    x -= s;
    y -= s;
    z -= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator-=(const vec<scal_type, 3>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator*=(const scal_type s)
{
    x *= s;
    y *= s;
    z *= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator*=(const vec<scal_type, 3>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator/=(const scal_type s)
{
    x /= s;
    y /= s;
    z /= s;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator/=(const vec<scal_type, 3>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator++(int)
{
    vec<scal_type, 3> tmp(*this);

    x += scal_type(1);
    y += scal_type(1);
    z += scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator++()
{
    x += scal_type(1);
    y += scal_type(1);
    z += scal_type(1);

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator--(int)
{
    vec<scal_type, 3> tmp(*this);

    x -= scal_type(1);
    y -= scal_type(1);
    z -= scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 3>& vec<scal_type, 3>::operator--()
{
    x -= scal_type(1);
    y -= scal_type(1);
    z -= scal_type(1);

    return (*this);
}

template<typename scal_type>
inline const vec<scal_type, 3>& vec<scal_type, 3>::operator-() const
{
    return (vec<scal_type, 3>(-x, -y, -z));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator+(const vec<scal_type,   3>& lhs,
                                         const vec<scal_type_r, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs.x + rhs.x,
                              lhs.y + rhs.y,
                              lhs.z + rhs.z));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator-(const vec<scal_type,   3>& lhs,
                                         const vec<scal_type_r, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs.x - rhs.x,
                              lhs.y - rhs.y,
                              lhs.z - rhs.z));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator*(const vec<scal_type,   3>& lhs,
                                         const vec<scal_type_r, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs.x * rhs.x,
                              lhs.y * rhs.y,
                              lhs.z * rhs.z));
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 3> operator*(const scal_type_l        lhs,
                                         const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs * rhs.x,
                              lhs * rhs.y,
                              lhs * rhs.z));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator*(const vec<scal_type, 3>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 3>(lhs.x * rhs,
                              lhs.y * rhs,
                              lhs.z * rhs));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator/(const vec<scal_type,   3>& lhs,
                                         const vec<scal_type_r, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs.x / rhs.x,
                              lhs.y / rhs.y,
                              lhs.z / rhs.z));
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 3> operator/(const scal_type_l        lhs,
                                         const vec<scal_type, 3>& rhs)
{
    return (vec<scal_type, 3>(lhs / rhs.x,
                              lhs / rhs.y,
                              lhs / rhs.z));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 3> operator/(const vec<scal_type, 3>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 3>(lhs.x / rhs,
                              lhs.y / rhs,
                              lhs.z / rhs));
}

} // namespace math
} // namespace scm
