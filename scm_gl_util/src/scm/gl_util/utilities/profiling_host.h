
#ifndef SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
#define SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED

#include <cassert>
#include <map>
#include <string>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/time/time_types.h>
#include <scm/core/time/accumulate_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/cl_core/cuda/cuda_fwd.h>
#include <scm/cl_core/opencl/opencl_fwd.h>

#include <scm/gl_core/render_device/render_device_fwd.h>

#include <scm/gl_util/utilities/utilities_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace util {

struct cpu_timer  {};
struct ogl_timer  {};
struct ocl_timer  {};
struct cuda_timer {};

class __scm_export(gl_util) profiling_host
{
public:
    typedef time::time_duration                                     duration_type;
    typedef time::accumulate_timer<time::high_res_timer>            cpu_timer_type;
    typedef std::map<std::string, cpu_timer_type>                   cpu_timer_map;
    typedef std::map<std::string, gl::accumulate_timer_query_ptr>   gl_timer_map;
    typedef std::map<std::string, cl::util::accum_timer_ptr>        cl_timer_map;
    typedef std::map<std::string, cu::util::accum_timer_ptr>        cu_timer_map;

public:
    profiling_host();
    virtual ~profiling_host();

    //template<typename timer_tag>
    //void
    //start(const std::string& tname) {
    //    assert(0);
    //}
    //template<>
    //void
    //start<cpu_timer>(const std::string& tname) {
    //    auto t = _cpu_timers.find(tname);
    //    if (t == _cpu_timers.end()) {
    //        t = _cpu_timers.insert(cpu_timer_map::value_type(tname, cpu_timer_type())).first;
    //    }
    //    _cpu_timers[tname].start();
    //}
    //template<>
    //void
    //start<ogl_timer>(const std::string& tname) {
    //    auto t = _gl_timers.find(tname);
    //    if (t == _gl_timers.end()) {

    //        t = _gl_timers.insert(cpu_timer_map::value_type(tname, cpu_timer_type())).first;
    //    }
    //    _cpu_timers[tname].start();
    //}

    void                    stop(const std::string& tname);

    void                    collect(const std::string& tname);
    void                    reset(const std::string& tname);
    
    duration_type           accumulated_duration(const std::string& tname) const;
    unsigned                accumulation_count(const std::string& tname) const;

    duration_type           average_duration(const std::string& tname) const;

protected:
    cpu_timer_map           _cpu_timers;
    gl_timer_map            _gl_timers;
    cl_timer_map            _cl_timers;
    cu_timer_map            _cu_timers;

}; // profiling_host

#include "profiling_host.inl"

} // namespace util
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif //SCM_GL_UTIL_PROFILING_HOST_H_INCLUDED
