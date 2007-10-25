
#include "time_query.h"

#include <cassert>

#include <scm/ogl/gl.h>
#include <scm/ogl.h>

namespace scm {
namespace gl {

time_query::time_query()
    : _id(0)
{
    glGenQueries(1, &_id);
}

time_query::~time_query()
{
    glDeleteQueries(1, &_id);
}

void time_query::start()
{
    glBeginQuery(GL_TIME_ELAPSED_EXT, _id);
}

void time_query::stop()
{
    glEndQuery(GL_TIME_ELAPSED_EXT);
}

void time_query::collect_result()
{
    static GLuint64EXT dur = 0;

    glGetQueryObjectui64vEXT(_id, GL_QUERY_RESULT, &dur);

    _duration = scm::time::nanosec(static_cast<scm::core::uint64_t>(dur));
}

bool time_query::is_supported()
{
    return (scm::ogl.get().is_supported("GL_EXT_timer_query"));
}

} // namespace gl
} // namespace scm
