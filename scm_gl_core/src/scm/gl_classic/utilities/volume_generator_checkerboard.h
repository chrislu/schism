
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED
#define VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED

#include <boost/scoped_array.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl_classic {

struct __scm_export(gl_core) volume_generator_checkerboard
{
    bool      generate_float_volume(unsigned /*dim_x*/,
                                    unsigned /*dim_y*/,
                                    unsigned /*dim_z*/,
                                    unsigned /*components*/,
                                    boost::scoped_array<float>& /*buffer*/);
}; // struct volume_generator_checkerboard

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // VOLUME_GENERATOR_CHECKERBOARD_H_INCLUDED
