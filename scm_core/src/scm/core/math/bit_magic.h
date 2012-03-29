
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef CORE_MATH_BIT_MAGIC_H_INCLUDED
#define CORE_MATH_BIT_MAGIC_H_INCLUDED

namespace scm {
namespace math {

// 0 1 2 3
// | \ / |
// |  X  |
// | / \ |
// 0 2 1 3
inline unsigned butterfly_1(unsigned x)
{
    static const unsigned b  = 1;
    static const unsigned ml = static_cast<unsigned>(0x4444444444444444u);
    static const unsigned mr = ml >> b;

    return    ((x & ml) >> b)
            | ((x & mr) << b)
            | (x & ~(ml | mr));
}

inline unsigned butterfly_2(unsigned x)
{
    static const unsigned b  = 2;
    static const unsigned ml = static_cast<unsigned>(0x3030303030303030u);
    static const unsigned mr = ml >> b;

    return    ((x & ml) >> b)
            | ((x & mr) << b)
            | (x & ~(ml | mr));
}

inline unsigned butterfly_4(unsigned x)
{
    static const unsigned b  = 4;
    static const unsigned ml = static_cast<unsigned>(0x0f000f000f000f00u);
    static const unsigned mr = ml >> b;

    return    ((x & ml) >> b)
            | ((x & mr) << b)
            | (x & ~(ml | mr));
}

inline unsigned butterfly_8(unsigned x)
{
    static const unsigned b  = 8;
    static const unsigned ml = static_cast<unsigned>(0x00ff000000ff0000u);
    static const unsigned mr = ml >> b;

    return    ((x & ml) >> b)
            | ((x & mr) << b)
            | (x & ~(ml | mr));
}

inline unsigned butterfly_16(unsigned x)
{
    static const unsigned b  = 16;
    static const unsigned ml = 0x0000ffff00000000u;
    static const unsigned mr = ml >> b;

    return    ((x & ml) >> b)
            | ((x & mr) << b)
            | (x & ~(ml | mr));
}

} // namespace math
} // namespace scm

#endif // CORE_MATH_BIT_MAGIC_H_INCLUDED
