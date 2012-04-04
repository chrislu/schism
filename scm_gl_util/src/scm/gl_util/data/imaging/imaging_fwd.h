
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_IMAGING_FWD_H_INCLUDED
#define SCM_GL_UTIL_IMAGING_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class texture_image_data;

typedef shared_ptr<texture_image_data>          texture_image_data_ptr;
typedef shared_ptr<texture_image_data const>    texture_image_data_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_IMAGING_FWD_H_INCLUDED
