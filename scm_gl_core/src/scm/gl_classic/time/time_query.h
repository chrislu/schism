
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TIME_QUERY_H_INCLUDED
#define TIME_QUERY_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/time/timer_interface.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) time_query : public scm::time::timer_interface
{
public:
    time_query();
    virtual ~time_query();

    void        start();
    void        stop();

    void        collect_result() const;

    static bool is_supported();

protected:

private:
    unsigned                _id;

    // not to be used
    void        intermediate_stop();
}; //class time_query

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TIME_QUERY_H_INCLUDED
