
#ifndef SCM_GL_CORE_QUERY_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_QUERY_OBJECTS_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class query;
class timer_query;
class transform_feedback_statistics_query;

typedef shared_ptr<query>                                       query_ptr;
typedef shared_ptr<query const>                                 query_cptr;
typedef shared_ptr<timer_query>                                 timer_query_ptr;
typedef shared_ptr<timer_query const>                           timer_query_cptr;
typedef shared_ptr<transform_feedback_statistics_query>         transform_feedback_statistics_query_ptr;
typedef shared_ptr<transform_feedback_statistics_query const>   transform_feedback_statistics_query_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_QUERY_OBJECTS_FWD_H_INCLUDED
