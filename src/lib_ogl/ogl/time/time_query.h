
#ifndef TIME_QUERY_H_INCLUDED
#define TIME_QUERY_H_INCLUDED

#include <scm_core/core/int_types.h>
#include <scm_core/time/timer_interface.h>

namespace gl {

class time_query : public scm::time::timer_interface
{
public:
    time_query();
    virtual ~time_query();

    void        start();
    void        stop();

    void        collect_result();

    static bool is_supported();

protected:

private:
    unsigned                _id;

}; //class time_query

} // namespace gl

#endif // TIME_QUERY_H_INCLUDED
