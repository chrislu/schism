
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "time_query.h"

#include <cassert>

#include <scm/gl_classic.h>
#include <scm/gl_classic/opengl.h>

namespace scm {
namespace gl_classic {

time_query::time_query()
  : _id(0),
    timer_interface(nano_seconds)
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

void time_query::intermediate_stop()
{
    // not implemented
    //glEndQuery(GL_TIME_ELAPSED_EXT);
}

void time_query::collect_result() const
{
    GLuint64EXT dur = 0;

    glGetQueryObjectui64vEXT(_id, GL_QUERY_RESULT, &dur);

    _duration = scm::time::nanosec(static_cast<scm::uint64>(dur));
}

bool time_query::is_supported()
{
    return (scm::opengl::get().is_supported("GL_EXT_timer_query"));
}

} // namespace gl_classic
} // namespace scm
