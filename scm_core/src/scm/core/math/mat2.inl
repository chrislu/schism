
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline mat<scal_type, 2, 2>::mat()
{
}

template<typename scal_type>
inline mat<scal_type, 2, 2>::mat(const mat<scal_type, 2, 2>& m)
  : m00(m.m00), m01(m.m01),
    m02(m.m02), m03(m.m03)
{
    //std::copy(m.data_array, m.data_array + 4, data_array);
}

//template<typename scal_type>
//inline mat<scal_type, 2, 2>::mat(const scal_type a[4])
//{
//    std::copy(a, a + 4, data_array);
//}

template<typename scal_type>
inline mat<scal_type, 2, 2>::mat(const scal_type a00, const scal_type a01,
                                 const scal_type a02, const scal_type a03)
  : m00(a00), m01(a01),
    m02(a02), m03(a03)
{
}

template<typename scal_type>
inline mat<scal_type, 2, 2>::mat(const vec<scal_type, 2>& c00,
                                 const vec<scal_type, 2>& c01)
{
    std::copy(c00.data_array, c00.data_array + 3, data_array);
    std::copy(c01.data_array, c01.data_array + 3, data_array + 2);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 2, 2>::mat(const mat<rhs_scal_t, 2, 2>& m)
  : m00(static_cast<scal_type>(m.m00)), m01(static_cast<scal_type>(m.m01)),
    m02(static_cast<scal_type>(m.m02)), m03(static_cast<scal_type>(m.m03))
{
}

// constants
template<typename scal_type>
const mat<scal_type, 2, 2>& mat<scal_type, 2, 2>::zero()
{
    static mat<scal_type, 2, 2> zero_(scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0));

    return (zero_);
}

template<typename scal_type>
const mat<scal_type, 2, 2>& mat<scal_type, 2, 2>::identity()
{
    static mat<scal_type, 2, 2> identity_(scal_type(1), scal_type(0),
                                          scal_type(0), scal_type(1));

    return (identity_);
}

// dtor
//template<typename scal_type>
//inline mat<scal_type, 2, 2>::~mat()
//{
//}

// swap
template<typename scal_type>
inline void mat<scal_type, 2, 2>::swap(mat<scal_type, 2, 2>& rhs)
{
    std::swap_ranges(data_array, data_array + 4, rhs.data_array);
}

// assign
template<typename scal_type>
inline mat<scal_type, 2, 2>& mat<scal_type, 2, 2>::operator=(const mat<scal_type, 2, 2>& rhs)
{
    std::copy(rhs.data_array, rhs.data_array + 4, data_array);

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 2, 2>& mat<scal_type, 2, 2>::operator=(const mat<rhs_scal_t, 2, 2>& rhs)
{
    for (unsigned i = 0; i < 4; ++i) {
        data_array[i] = rhs.data_array[i];
    }

    return (*this);
}

// index
template<typename scal_type>
inline scal_type& mat<scal_type, 2, 2>::operator[](const int i)
{
    assert(i < 4);

    return (data_array[i]);
}

template<typename scal_type>
inline scal_type  mat<scal_type, 2, 2>::operator[](const int i) const
{
    assert(i < 4);

    return (data_array[i]);
}

template<typename scal_type>
inline vec<scal_type, 2> mat<scal_type, 2, 2>::column(const int i) const
{
    return (vec<scal_type, 2>(data_array[i * 2],
                              data_array[i * 2 + 1]));
}

template<typename scal_type>
inline vec<scal_type, 2> mat<scal_type, 2, 2>::row(const int i) const
{
    return (vec<scal_type, 2>(data_array[i],
                              data_array[i + 2]));
}

} // namespace math
} // namespace scm
