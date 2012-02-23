
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_WAVEFRONT_OBJ_LOADER_H_INCLUDED
#define SCM_GL_UTIL_WAVEFRONT_OBJ_LOADER_H_INCLUDED

#endif // SCM_GL_UTIL_WAVEFRONT_OBJ_LOADER_H_INCLUDED
#ifndef SCM_DATA_OBJ_LOADER_H_INCLUDED
#define SCM_DATA_OBJ_LOADER_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {
namespace util {

struct wavefront_model;

bool __scm_export(gl_util) open_obj_file(const std::string& filename, wavefront_model& /*out_obj*/);

} // namespace util
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_WAVEFRONT_OBJ_LOADER_H_INCLUDED
