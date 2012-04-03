
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED
#define SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class accum_timer_query;
typedef shared_ptr<accum_timer_query>          accum_timer_query_ptr;
typedef shared_ptr<accum_timer_query const>    accum_timer_query_cptr;

class coordinate_cross;
typedef shared_ptr<coordinate_cross>                coordinate_cross_ptr;
typedef shared_ptr<coordinate_cross const>          coordinate_cross_cptr;

class geometry_highlight;
typedef shared_ptr<geometry_highlight>              geometry_highlight_ptr;
typedef shared_ptr<geometry_highlight const>        geometry_highlight_cptr;

class texture_output;
typedef shared_ptr<texture_output>                  texture_output_ptr;
typedef shared_ptr<texture_output const>            texture_output_cptr;

namespace util {

class  profiling_host;
struct profiling_result;
typedef shared_ptr<profiling_host>                  profiling_host_ptr;
typedef shared_ptr<profiling_host const>            profiling_host_cptr;

class overlay_text_output;
typedef shared_ptr<overlay_text_output>             overlay_text_output_ptr;
typedef shared_ptr<overlay_text_output const>       overlay_text_output_cptr;

} // namespace util

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED
