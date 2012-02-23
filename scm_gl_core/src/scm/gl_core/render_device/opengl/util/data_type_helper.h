
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OPENGL_DATA_TYPE_HELPER_H_INCLUDED
#define SCM_GL_CORE_OPENGL_DATA_TYPE_HELPER_H_INCLUDED

#include <scm/gl_core/data_types.h>

namespace scm {
namespace gl {
namespace util {

data_type   from_gl_data_type(unsigned gl_type);
bool        is_sampler_type(unsigned gl_type);
bool        is_image_type(unsigned gl_type);
bool        is_vaild_index_type(const data_type d);

unsigned    gl_base_type(const data_type d);


} // namespace util
} // namespace gl
} // namespace scm

#include "data_type_helper.inl"

#endif // SCM_GL_CORE_OPENGL_DATA_TYPE_HELPER_H_INCLUDED
