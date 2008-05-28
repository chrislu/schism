
#ifndef TIME_QUERY_H_INCLUDED
#define TIME_QUERY_H_INCLUDED

#include <scm/core/int_types.h>
#include <scm/core/time/timer_interface.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) time_query : public scm::time::timer_interface
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

}; //class time_query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TIME_QUERY_H_INCLUDED
