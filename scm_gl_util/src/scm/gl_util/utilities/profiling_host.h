
#ifndef SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
#define SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED

#include <iosfwd>
#include <cassert>
#include <map>
#include <string>
#include <utility>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/time/accum_timer.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace cl {

class Event;

} // namespace cl

namespace scm {
namespace cu {

class cuda_command_stream;

typedef scm::shared_ptr<cuda_command_stream>        cuda_command_stream_ptr;
typedef scm::shared_ptr<cuda_command_stream const>  cuda_command_stream_cptr;

} // namespace cu

namespace gl {
namespace util {

class __scm_export(gl_util) profiling_host
{
public:
    typedef time::accum_timer_base::duration_type               duration_type;

protected:
    enum timer_type {
        CPU_TIMER = 0x00,
        GL_TIMER,
        CU_TIMER,
        CL_TIMER
    };
    typedef shared_ptr<time::accum_timer_base>  timer_ptr;
    struct timer_instance {
        timer_instance(timer_type t, time::accum_timer_base* tm) : _type(t), _timer(tm), _time(duration_type()) {}
        timer_type      _type;
        timer_ptr       _timer;
        duration_type   _time;
    };
    typedef std::map<std::string, timer_instance> timer_map;

public:
    profiling_host();
    virtual ~profiling_host();

    void                    cpu_start(const std::string& tname);
    void                    gl_start(const std::string& tname, const render_context_ptr& context);
    void                    cu_start(const std::string& tname, const cu::cuda_command_stream_ptr& cu_stream);
    ::cl::Event*const       cl_start(const std::string& tname);

    void                    stop(const std::string& tname);
    duration_type           time(const std::string& tname) const;

    void                    update(int interval = 100);

    void                    collect_all();
    void                    reset_all();

    void                    collect(const std::string& tname);
    void                    reset(const std::string& tname);
    
    duration_type           accumulated_duration(const std::string& tname) const;
    unsigned                accumulation_count(const std::string& tname) const;

    duration_type           average_duration(const std::string& tname) const;

    std::string             timer_prefix_string(const std::string& tname) const;
    std::string             timer_prefix_string(timer_type ttype) const;
    std::string             timer_type_string(timer_type ttype) const;

protected:
    timer_ptr               find_timer(const std::string& tname) const;

protected:
    timer_map               _timers;
    int                     _update_interval;

}; // profiling_host

struct __scm_export(gl_util) profiling_result
{
    enum time_unit {
        sec,
        msec,
        usec,
        nsec
    };
    enum throughput_unit {
        Bps,
        KiBps,
        MiBps,
        GiBps
    };
    profiling_result(const profiling_host_cptr& host,
                     const std::string&         tname,
                           time_unit            tunit = msec);
    profiling_result(const profiling_host_cptr& host,
                     const std::string&         tname,
                           scm::size_t          dsize,
                           time_unit            tunit = msec,
                           throughput_unit      d_unit = MiBps);

    std::string         unit_string() const;
    std::string         throughput_string() const;
    double              time() const;
    double              throughput() const;

    profiling_host_cptr _phost;
    std::string         _tname;
    time_unit           _tunit;
    scm::size_t         _dsize;
    throughput_unit     _dunit;
}; // struct profiling_result

__scm_export(gl_util) std::ostream& operator<<(std::ostream& os, const profiling_result& pres);

} // namespace util
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
