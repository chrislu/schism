
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DATA_FORMAT_HELPER_H_INCLUDED
#define SCM_GL_CORE_DATA_FORMAT_HELPER_H_INCLUDED

#include <scm/gl_core/data_formats.h>

namespace scm {
namespace gl {
namespace util {

unsigned gl_internal_format(const data_format d);
unsigned gl_base_format(const data_format d);
unsigned gl_base_type(const data_format d);

} // namespace util
} // namespace gl
} // namespace scm

#include "data_format_helper.inl"

#endif // SCM_GL_CORE_DATA_FORMAT_HELPER_H_INCLUDED
