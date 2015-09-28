
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
#define SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED

#include <iosfwd>
#include <cassert>
#include <map>
#include <string>
#include <utility>

#include <scm/config.h>
#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/time/accum_timer_base.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace cl {

class Event;

} // namespace cl

namespace scm {
#if SCM_ENABLE_CUDA_CL_SUPPORT
namespace cu {

class cuda_command_stream;

typedef scm::shared_ptr<cuda_command_stream>        cuda_command_stream_ptr;
typedef scm::shared_ptr<cuda_command_stream const>  cuda_command_stream_cptr;

} // namespace cu
#endif
namespace gl {
namespace util {

class __scm_export(gl_util) profiling_host
{
public:
    typedef time::accum_timer_base::nanosec_type        nanosec_type;
    typedef shared_ptr<time::accum_timer_base>  timer_ptr;

protected:
    enum timer_type {
        CPU_TIMER = 0x00,
        GL_TIMER,
        CU_TIMER,
        CL_TIMER
    };
    struct timer_instance {
        timer_instance(timer_type t, time::accum_timer_base* tm) : _type(t), _timer(tm), _time(0) {}
        timer_type      _type;
        timer_ptr       _timer;
        nanosec_type    _time;
    };
    typedef std::map<std::string, timer_instance> timer_map;

public:
    profiling_host();
    virtual ~profiling_host();

    bool                    enabled() const;
    void                    enabled(bool e);

    void                    cpu_start(const std::string& tname);
    void                    gl_start(const std::string& tname, const render_context_ptr& context);
#if SCM_ENABLE_CUDA_CL_SUPPORT
    void                    cu_start(const std::string& tname, const cu::cuda_command_stream_ptr& cu_stream);
    ::cl::Event*const       cl_start(const std::string& tname);
#endif
    void                    stop(const std::string& tname) const;
    nanosec_type            time(const std::string& tname) const;

    void                    update(int interval = 100);
    void                    force_update();

    void                    collect_all();
    void                    force_collect_all();
    void                    reset_all();

    void                    collect(const std::string& tname);
    void                    reset(const std::string& tname);
    
    nanosec_type            accumulated_time(const std::string& tname) const;
    unsigned                accumulation_count(const std::string& tname) const;

    nanosec_type            average_time(const std::string& tname) const;

    std::string             timer_prefix_string(const std::string& tname) const;
    std::string             timer_prefix_string(timer_type ttype) const;
    std::string             timer_type_string(timer_type ttype) const;

    timer_ptr               find_timer(const std::string& tname) const;
protected:

protected:
    bool                    _enabled;
    timer_map               _timers;
    int                     _update_interval;

}; // profiling_host

class __scm_export(gl_util) scoped_timer
{
public:
    scoped_timer(profiling_host& phost, const std::string& tname);
    scoped_timer(profiling_host& phost, const std::string& tname, const render_context_ptr& context);
#if SCM_ENABLE_CUDA_CL_SUPPORT
    scoped_timer(profiling_host& phost, const std::string& tname, const cu::cuda_command_stream_ptr& cu_stream);
#endif
    ~scoped_timer();

private:
    const profiling_host& _phost;
    const std::string     _tname;

private: // declared, never defined
    scoped_timer(const scoped_timer&);
    const scoped_timer& operator=(const scoped_timer&);
};

struct __scm_export(gl_util) profiling_result
{
    profiling_result(const profiling_host_cptr& host,
                     const std::string&         tname,
                           time::time_io        unit   = time::time_io(time::time_io::msec));
    profiling_result(const profiling_host_cptr& host,
                     const std::string&         tname,
                           scm::size_t          dsize,
                           time::time_io        unit   = time::time_io(time::time_io::msec, time::time_io::MiBps));

    std::string         unit_string() const;
    std::string         throughput_string() const;
    double              time() const;
    double              throughput() const;

    profiling_host_cptr _phost;
    std::string         _tname;
    scm::size_t         _dsize;
    time::time_io       _unit;
}; // struct profiling_result

__scm_export(gl_util) std::ostream& operator<<(std::ostream& os, const profiling_result& pres);

} // namespace util
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
