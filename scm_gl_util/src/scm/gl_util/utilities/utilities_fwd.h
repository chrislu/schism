
#ifndef SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED
#define SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class accumulate_timer_query;
class coordinate_cross;
class geometry_highlight;
class texture_output;

typedef shared_ptr<accumulate_timer_query>          accumulate_timer_query_ptr;
typedef shared_ptr<accumulate_timer_query const>    accumulate_timer_query_cptr;
typedef shared_ptr<coordinate_cross>                coordinate_cross_ptr;
typedef shared_ptr<coordinate_cross const>          coordinate_cross_cptr;
typedef shared_ptr<geometry_highlight>              geometry_highlight_ptr;
typedef shared_ptr<geometry_highlight const>        geometry_highlight_cptr;
typedef shared_ptr<texture_output>                  texture_output_ptr;
typedef shared_ptr<texture_output const>            texture_output_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_UTILITIES_FWD_H_INCLUDED
