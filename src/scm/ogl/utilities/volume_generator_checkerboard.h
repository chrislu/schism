
#ifndef VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED
#define VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED

#include <boost/scoped_array.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(ogl) volume_generator_checkerboard
{
    bool      generate_float_volume(unsigned /*dim_x*/,
                                    unsigned /*dim_y*/,
                                    unsigned /*dim_z*/,
                                    unsigned /*components*/,
                                    boost::scoped_array<float>& /*buffer*/);
}; // struct volume_generator_checkerboard

} // namespace gl
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED
