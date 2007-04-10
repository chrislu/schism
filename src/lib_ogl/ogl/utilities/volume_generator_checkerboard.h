
#ifndef VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED
#define VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED

#include <boost/scoped_array.hpp>

namespace gl
{
    struct volume_generator_checkerboard
    {
        bool      generate_float_volume(unsigned /*dim_x*/,
                                        unsigned /*dim_y*/,
                                        unsigned /*dim_z*/,
                                        unsigned /*components*/,
                                        boost::scoped_array<float>& /*buffer*/);
    }; // struct volume_generator_checkerboard
} // namespace gl

#endif // VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED



