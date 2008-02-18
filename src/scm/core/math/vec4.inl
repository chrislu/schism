
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

// assign
template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator=(const vec<scal_type, 4>& rhs)
{
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
inline vec<scal_type, 4>& vec<scal_type, 4>::operator++(int)
{
    vec<scal_type, 4> tmp(*this);

    x += scal_type(1);
    y += scal_type(1);
    z += scal_type(1);
    w += scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator++()
{
    x += scal_type(1);
    y += scal_type(1);
    z += scal_type(1);
    w += scal_type(1);

    return (*this);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator--(int)
{
    vec<scal_type, 4> tmp(*this);

    x -= scal_type(1);
    y -= scal_type(1);
    z -= scal_type(1);
    w -= scal_type(1);

    return (tmp);
}

template<typename scal_type>
inline vec<scal_type, 4>& vec<scal_type, 4>::operator--()
{
    x -= scal_type(1);
    y -= scal_type(1);
    z -= scal_type(1);
    w -= scal_type(1);

    return (*this);
}

template<typename scal_type>
inline const vec<scal_type, 4>& vec<scal_type, 4>::operator-() const
{
    return (vec<scal_type, 4>(-x, -y, -z, -w));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator+(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs.x + rhs.x,
                              lhs.y + rhs.y,
                              lhs.z + rhs.z,
                              lhs.w + rhs.w));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator-(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs.x - rhs.x,
                              lhs.y - rhs.y,
                              lhs.z - rhs.z,
                              lhs.w - rhs.w));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator*(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs.x * rhs.x,
                              lhs.y * rhs.y,
                              lhs.z * rhs.z,
                              lhs.w * rhs.w));
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 4> operator*(const scal_type_l        lhs,
                                         const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs * rhs.x,
                              lhs * rhs.y,
                              lhs * rhs.z,
                              lhs * rhs.w));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator*(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs.x * rhs,
                              lhs.y * rhs,
                              lhs.z * rhs,
                              lhs.w * rhs));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator/(const vec<scal_type,   4>& lhs,
                                         const vec<scal_type_r, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs.x / rhs.x,
                              lhs.y / rhs.y,
                              lhs.z / rhs.z,
                              lhs.w / rhs.w));
}

template<typename scal_type, typename scal_type_l>
inline const vec<scal_type, 4> operator/(const scal_type_l        lhs,
                                         const vec<scal_type, 4>& rhs)
{
    return (vec<scal_type, 4>(lhs / rhs.x,
                              lhs / rhs.y,
                              lhs / rhs.z,
                              lhs / rhs.w));
}

template<typename scal_type, typename scal_type_r>
inline const vec<scal_type, 4> operator/(const vec<scal_type, 4>& lhs,
                                         const scal_type_r        rhs)
{
    return (vec<scal_type, 4>(lhs.x / rhs,
                              lhs.y / rhs,
                              lhs.z / rhs,
                              lhs.w / rhs));
}

} // namespace math
} // namespace scm
