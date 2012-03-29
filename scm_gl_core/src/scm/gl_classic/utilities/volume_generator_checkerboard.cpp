
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_generator_checkerboard.h"

#include <scm/core/math/math.h>
#include <exception>
#include <new>

namespace scm {
namespace gl_classic {

bool volume_generator_checkerboard::generate_float_volume(unsigned dim_x,
                                                          unsigned dim_y,
                                                          unsigned dim_z,
                                                          unsigned components,
                                                          boost::scoped_array<float>& buffer)

{
    if (dim_x < 1 || dim_y < 1 || dim_z < 1 || components < 1) {
        return (false);
    }

    try {
        buffer.reset(new float[dim_x * dim_y * dim_z * components]);
    }
    catch (std::bad_alloc&) {
        return (false);
    }

    float val;
    unsigned offset_dst;

    for (unsigned z = 0; z < dim_z; z++) {
        for (unsigned y = 0; y < dim_y; y++) {
            for (unsigned x = 0; x < dim_x; x++) {
                val = float(scm::math::sign(-int((x+y+z) % 2)));
                offset_dst =   x * components
                             + y * dim_x * components
                             + z * dim_x * dim_y * components;

                for (unsigned c = 0; c < components; c++) {
                    buffer[offset_dst + c] = val;
                }
            }
        }
    }

    return (true);
}

} // namespace gl_classic
} // namespace scm
