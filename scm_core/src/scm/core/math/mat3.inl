
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline mat<scal_type, 3, 3>::mat()
{
}

template<typename scal_type>
inline mat<scal_type, 3, 3>::mat(const mat<scal_type, 3, 3>& m)
  : m00(m.m00), m01(m.m01), m02(m.m02),
    m03(m.m03), m04(m.m04), m05(m.m05),
    m06(m.m06), m07(m.m07), m08(m.m08)
{
    //std::copy(m.data_array, m.data_array + 9, data_array);
}

//template<typename scal_type>
//inline mat<scal_type, 3, 3>::mat(const scal_type a[9])
//{
//    std::copy(a, a + 9, data_array);
//}

template<typename scal_type>
inline mat<scal_type, 3, 3>::mat(const scal_type a00, const scal_type a01, const scal_type a02,
                                 const scal_type a03, const scal_type a04, const scal_type a05,
                                 const scal_type a06, const scal_type a07, const scal_type a08)
  : m00(a00), m01(a01), m02(a02),
    m03(a03), m04(a04), m05(a05),
    m06(a06), m07(a07), m08(a08)
{
}

template<typename scal_type>
inline mat<scal_type, 3, 3>::mat(const vec<scal_type, 3>& c00,
                                 const vec<scal_type, 3>& c01,
                                 const vec<scal_type, 3>& c02)
{
    std::copy(c00.data_array, c00.data_array + 3, data_array);
    std::copy(c01.data_array, c01.data_array + 3, data_array + 3);
    std::copy(c02.data_array, c02.data_array + 3, data_array + 6);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 3, 3>::mat(const mat<rhs_scal_t, 3, 3>& m)
  : m00(static_cast<scal_type>(m.m00)), m01(static_cast<scal_type>(m.m01)), m02(static_cast<scal_type>(m.m02)),
    m03(static_cast<scal_type>(m.m03)), m04(static_cast<scal_type>(m.m04)), m05(static_cast<scal_type>(m.m05)),
    m06(static_cast<scal_type>(m.m06)), m07(static_cast<scal_type>(m.m07)), m08(static_cast<scal_type>(m.m08))
{
}

// constants
template<typename scal_type>
const mat<scal_type, 3, 3>& mat<scal_type, 3, 3>::zero()
{
    static mat<scal_type, 3, 3> zero_(scal_type(0), scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0), scal_type(0));

    return (zero_);
}

template<typename scal_type>
const mat<scal_type, 3, 3>& mat<scal_type, 3, 3>::identity()
{
    static mat<scal_type, 3, 3> identity_(scal_type(1), scal_type(0), scal_type(0),
                                          scal_type(0), scal_type(1), scal_type(0),
                                          scal_type(0), scal_type(0), scal_type(1));

    return (identity_);
}

// dtor
//template<typename scal_type>
//inline mat<scal_type, 3, 3>::~mat()
//{
//}

// swap
template<typename scal_type>
inline void mat<scal_type, 3, 3>::swap(mat<scal_type, 3, 3>& rhs)
{
    std::swap_ranges(data_array, data_array + 9, rhs.data_array);
}

// assign
template<typename scal_type>
inline mat<scal_type, 3, 3>& mat<scal_type, 3, 3>::operator=(const mat<scal_type, 3, 3>& rhs)
{
    std::copy(rhs.data_array, rhs.data_array + 9, data_array);

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 3, 3>& mat<scal_type, 3, 3>::operator=(const mat<rhs_scal_t, 3, 3>& rhs)
{
    for (unsigned i = 0; i < 9; ++i) {
        data_array[i] = rhs.data_array[i];
    }

    return (*this);
}

// index
template<typename scal_type>
inline scal_type& mat<scal_type, 3, 3>::operator[](const int i)
{
    assert(i < 9);

    return (data_array[i]);
}

template<typename scal_type>
inline scal_type  mat<scal_type, 3, 3>::operator[](const int i) const
{
    assert(i < 9);

    return (data_array[i]);
}

template<typename scal_type>
inline vec<scal_type, 3> mat<scal_type, 3, 3>::column(const int i) const
{
    return (vec<scal_type, 3>(data_array[i * 3],
                              data_array[i * 3 + 1],
                              data_array[i * 3 + 2]));
}

template<typename scal_type>
inline vec<scal_type, 3> mat<scal_type, 3, 3>::row(const int i) const
{
    return (vec<scal_type, 3>(data_array[i],
                              data_array[i + 3],
                              data_array[i + 6]));
}

} // namespace math
} // namespace scm
