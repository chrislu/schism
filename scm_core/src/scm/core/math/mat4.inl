
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>

namespace scm {
namespace math {

// ctors
template<typename scal_type>
inline mat<scal_type, 4, 4>::mat()
{
}

template<typename scal_type>
inline mat<scal_type, 4, 4>::mat(const mat<scal_type, 4, 4>& m)
  : m00(m.m00), m01(m.m01), m02(m.m02), m03(m.m03),
    m04(m.m04), m05(m.m05), m06(m.m06), m07(m.m07),
    m08(m.m08), m09(m.m09), m10(m.m10), m11(m.m11),
    m12(m.m12), m13(m.m13), m14(m.m14), m15(m.m15)
{
    //std::copy(m.data_array, m.data_array + 16, data_array);
}

//template<typename scal_type>
//inline mat<scal_type, 4, 4>::mat(const scal_type a[16])
//{
//    std::copy(a, a + 16, data_array);
//}

template<typename scal_type>
inline mat<scal_type, 4, 4>::mat(const scal_type a00, const scal_type a01, const scal_type a02, const scal_type a03,
                                 const scal_type a04, const scal_type a05, const scal_type a06, const scal_type a07,
                                 const scal_type a08, const scal_type a09, const scal_type a10, const scal_type a11,
                                 const scal_type a12, const scal_type a13, const scal_type a14, const scal_type a15)  
  : m00(a00), m01(a01), m02(a02), m03(a03),
    m04(a04), m05(a05), m06(a06), m07(a07),
    m08(a08), m09(a09), m10(a10), m11(a11),
    m12(a12), m13(a13), m14(a14), m15(a15)
{
}

template<typename scal_type>
inline mat<scal_type, 4, 4>::mat(const vec<scal_type, 4>& c00,
                                 const vec<scal_type, 4>& c01,
                                 const vec<scal_type, 4>& c02,
                                 const vec<scal_type, 4>& c03)
{
    std::copy(c00.data_array, c00.data_array + 4, data_array);
    std::copy(c01.data_array, c01.data_array + 4, data_array + 4);
    std::copy(c02.data_array, c02.data_array + 4, data_array + 8);
    std::copy(c03.data_array, c03.data_array + 4, data_array + 12);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 4, 4>::mat(const mat<rhs_scal_t, 4, 4>& m)
  : m00(static_cast<scal_type>(m.m00)), m01(static_cast<scal_type>(m.m01)), m02(static_cast<scal_type>(m.m02)), m03(static_cast<scal_type>(m.m03)),
    m04(static_cast<scal_type>(m.m04)), m05(static_cast<scal_type>(m.m05)), m06(static_cast<scal_type>(m.m06)), m07(static_cast<scal_type>(m.m07)),
    m08(static_cast<scal_type>(m.m08)), m09(static_cast<scal_type>(m.m09)), m10(static_cast<scal_type>(m.m10)), m11(static_cast<scal_type>(m.m11)),
    m12(static_cast<scal_type>(m.m12)), m13(static_cast<scal_type>(m.m13)), m14(static_cast<scal_type>(m.m14)), m15(static_cast<scal_type>(m.m15))
{
}

// constants
template<typename scal_type>
const mat<scal_type, 4, 4>& mat<scal_type, 4, 4>::zero()
{
    static mat<scal_type, 4, 4> zero_(scal_type(0), scal_type(0), scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0), scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0), scal_type(0), scal_type(0),
                                      scal_type(0), scal_type(0), scal_type(0), scal_type(0));

    return (zero_);
}

template<typename scal_type>
const mat<scal_type, 4, 4>& mat<scal_type, 4, 4>::identity()
{
    static mat<scal_type, 4, 4> identity_(scal_type(1), scal_type(0), scal_type(0), scal_type(0),
                                          scal_type(0), scal_type(1), scal_type(0), scal_type(0),
                                          scal_type(0), scal_type(0), scal_type(1), scal_type(0),
                                          scal_type(0), scal_type(0), scal_type(0), scal_type(1));

    return (identity_);
}

// dtor
//template<typename scal_type>
//inline mat<scal_type, 4, 4>::~mat()
//{
//}

// swap
template<typename scal_type>
inline void mat<scal_type, 4, 4>::swap(mat<scal_type, 4, 4>& rhs)
{
    std::swap_ranges(data_array, data_array + 16, rhs.data_array);
}

// assign
template<typename scal_type>
inline mat<scal_type, 4, 4>& mat<scal_type, 4, 4>::operator=(const mat<scal_type, 4, 4>& rhs)
{
    std::copy(rhs.data_array, rhs.data_array + 16, data_array);

    return (*this);
}

template<typename scal_type>
template<typename rhs_scal_t>
inline mat<scal_type, 4, 4>& mat<scal_type, 4, 4>::operator=(const mat<rhs_scal_t, 4, 4>& rhs)
{
    for (unsigned i = 0; i < 16; ++i) {
        data_array[i] = rhs.data_array[i];
    }

    return (*this);
}

// index
template<typename scal_type>
inline scal_type& mat<scal_type, 4, 4>::operator[](const int i)
{
    assert(i < 16);

    return (data_array[i]);
}

template<typename scal_type>
inline scal_type  mat<scal_type, 4, 4>::operator[](const int i) const
{
    assert(i < 16);

    return (data_array[i]);
}

template<typename scal_type>
inline vec<scal_type, 4> mat<scal_type, 4, 4>::column(const int i) const
{
    return (vec<scal_type, 4>(data_array[i * 4],
                              data_array[i * 4 + 1],
                              data_array[i * 4 + 2],
                              data_array[i * 4 + 3]));
}

template<typename scal_type>
inline vec<scal_type, 4> mat<scal_type, 4, 4>::row(const int i) const
{
    return (vec<scal_type, 4>(data_array[i],
                              data_array[i + 4],
                              data_array[i + 8],
                              data_array[i + 12]));
}

} // namespace math
} // namespace scm
