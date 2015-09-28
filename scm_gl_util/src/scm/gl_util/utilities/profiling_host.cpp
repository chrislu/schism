
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "profiling_host.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/io/ios_state.hpp>

#include <scm/config.h>
#include <scm/cl_core/opencl/CL/cl.hpp>

#include <scm/core/time/cpu_accum_timer.h>

#if SCM_ENABLE_CUDA_CL_SUPPORT
#include <scm/cl_core/cuda/accum_timer.h>
#include <scm/cl_core/cuda/command_stream.h>
#include <scm/cl_core/opencl/opencl_fwd.h>
#include <scm/cl_core/opencl/accum_timer.h>
#endif

#include <scm/gl_core/render_device.h>
#include <scm/gl_util/utilities/accum_timer_query.h>

namespace {

typedef scm::time::cpu_accum_timer  cpu_accum_timer;
typedef scm::gl::accum_timer_query  gl_accum_timer;

#if SCM_ENABLE_CUDA_CL_SUPPORT
typedef scm::cu::util::accum_timer  cu_accum_timer;
typedef scm::cl::util::accum_timer  cl_accum_timer;
#endif

struct null_deleter { void operator()(void const *) const {} };

}

namespace scm {
namespace gl {
namespace util {

profiling_host::profiling_host()
  : _enabled(false)
  , _update_interval(0)
{
}

profiling_host::~profiling_host()
{
    _timers.clear();
}

bool
profiling_host::enabled() const
{
    return _enabled;
}

void
profiling_host::enabled(bool e)
{
    _enabled = e;
}

void
profiling_host::cpu_start(const std::string& tname)
{
    if (_enabled) {
        cpu_accum_timer* t  = 0;
        auto             ti = _timers.find(tname);
        if (ti == _timers.end()) {
            t  = new cpu_accum_timer();
            _timers.insert(timer_map::value_type(tname, timer_instance(CPU_TIMER, t)));
        }
        else {
            if (CPU_TIMER != ti->second._type) {
                std::stringstream os;
                os << "profiling_host::cpu_start() "
                   << "timer with name '" << tname << "' already exists with different type [" << timer_type_string(ti->second._type) << "].";
                throw std::runtime_error(os.str());
            }

            t = dynamic_cast<cpu_accum_timer*>(ti->second._timer.get());
        }

        assert(0 != t);

        t->start();
    }
}

void
profiling_host::gl_start(const std::string& tname, const render_context_ptr& context)
{
    if (_enabled) {
        gl_accum_timer*  t  = 0;
        auto             ti = _timers.find(tname);
        if (ti == _timers.end()) {
            // EVIL!!!111einseinself
            render_device_ptr d(&(context->parent_device()), null_deleter());
            t = new gl_accum_timer(d);
            _timers.insert(timer_map::value_type(tname, timer_instance(GL_TIMER, t)));
        }
        else {
            if (GL_TIMER != ti->second._type) {
                std::stringstream os;
                os << "profiling_host::gl_start() "
                   << "timer with name '" << tname << "' already exists with different type [" << timer_type_string(ti->second._type) << "].";
                throw std::runtime_error(os.str());
            }

            t = dynamic_cast<gl_accum_timer*>(ti->second._timer.get());
        }

        assert(0 != t);

        t->start(context);
    }
}

#if SCM_ENABLE_CUDA_CL_SUPPORT
void
profiling_host::cu_start(const std::string& tname, const cu::cuda_command_stream_ptr& cu_stream)
{
    if (_enabled) {
        cu_accum_timer* t  = 0;
        auto            ti = _timers.find(tname);
        if (ti == _timers.end()) {
            t  = new cu_accum_timer();
            _timers.insert(timer_map::value_type(tname, timer_instance(CU_TIMER, t)));
        }
        else {
            if (CU_TIMER != ti->second._type) {
                std::stringstream os;
                os << "profiling_host::cu_start() "
                   << "timer with name '" << tname << "' already exists with different type [" << timer_type_string(ti->second._type) << "].";
                throw std::runtime_error(os.str());
            }

            t = dynamic_cast<cu_accum_timer*>(ti->second._timer.get());
        }

        assert(0 != t);

        if (cu_stream) {
            t->start(cu_stream->stream());
        }
        else {
            t->start();
        }
    }
}

::cl::Event*const
profiling_host::cl_start(const std::string& tname)
{
    if (_enabled) {
        cl_accum_timer* t  = 0;
        auto            ti = _timers.find(tname);
        if (ti == _timers.end()) {
            t  = new cl_accum_timer();
            _timers.insert(timer_map::value_type(tname, timer_instance(CL_TIMER, t)));
        }
        else {
            if (CL_TIMER != ti->second._type) {
                std::stringstream os;
                os << "profiling_host::cl_start() "
                   << "timer with name '" << tname << "' already exists with different type [" << timer_type_string(ti->second._type) << "].";
                throw std::runtime_error(os.str());
            }

            t = dynamic_cast<cl_accum_timer*>(ti->second._timer.get());
        }

        assert(0 != t);

        return t->event();
    }
    else {
        return 0;
    }
}
#endif

void
profiling_host::stop(const std::string& tname) const
{
    if (_enabled) {
        timer_ptr t = find_timer(tname);
        if (t) {
            t->stop();
        }
    }
}

profiling_host::nanosec_type
profiling_host::time(const std::string& tname) const
{
    auto ti = _timers.find(tname);
    if (ti != _timers.end()) {
        return ti->second._time;
    }

    return 0;
}

void
profiling_host::update(int interval)
{
    if (_enabled) {
        using namespace std;

        collect_all();
        ++_update_interval;

        if (_update_interval >= interval) {
            _update_interval = 0;

            for_each(_timers.begin(), _timers.end(), [](timer_map::value_type& t) -> void {
                t.second._timer->update(0);
                t.second._time = t.second._timer->average_time();
            });

            reset_all();
        }
    }
}

void
profiling_host::force_update()
{
    if (_enabled) {
        using namespace std;

        force_collect_all();

        _update_interval = 0;

        for_each(_timers.begin(), _timers.end(), [](timer_map::value_type& t) -> void {
            t.second._timer->update(0);
            t.second._time = t.second._timer->average_time();
        });

        reset_all();
    }
}

void
profiling_host::collect_all()
{
    using namespace std;
    for_each(_timers.begin(), _timers.end(), [](timer_map::value_type& t) -> void {
        t.second._timer->collect();
    });
}

void
profiling_host::force_collect_all()
{
    using namespace std;
    for_each(_timers.begin(), _timers.end(), [](timer_map::value_type& t) -> void {
        t.second._timer->force_collect();
    });
}

void
profiling_host::reset_all()
{
    using namespace std;
    for_each(_timers.begin(), _timers.end(), [](timer_map::value_type& t) -> void {
        t.second._timer->reset();
    });
}

void
profiling_host::collect(const std::string& tname)
{
    timer_ptr t = find_timer(tname);
    if (t) {
        t->collect();
    }
}

void
profiling_host::reset(const std::string& tname)
{
    timer_ptr t = find_timer(tname);
    if (t) {
        t->reset();
    }
}

profiling_host::nanosec_type
profiling_host::accumulated_time(const std::string& tname) const
{
    timer_ptr t = find_timer(tname);
    if (t) {
        return t->accumulated_time();
    }

    return 0;
}

unsigned
profiling_host::accumulation_count(const std::string& tname) const
{
    timer_ptr t = find_timer(tname);
    if (t) {
        return t->accumulation_count();
    }

    return 0u;
}

profiling_host::nanosec_type
profiling_host::average_time(const std::string& tname) const
{
    timer_ptr t = find_timer(tname);
    if (t) {
        return t->average_time();
    }

    return 0;
}

std::string
profiling_host::timer_type_string(timer_type ttype) const
{
    std::string r;

    switch (ttype) {
        case CPU_TIMER: r.assign("CPU_TIMER"); break;
        case GL_TIMER:  r.assign("GL_TIMER"); break;
#if SCM_ENABLE_CUDA_CL_SUPPORT
        case CU_TIMER:  r.assign("CU_TIMER"); break;
        case CL_TIMER:  r.assign("CL_TIMER"); break;
#endif
        default:        r.assign("unknown");
    }

    return r;
}

std::string
profiling_host::timer_prefix_string(const std::string& tname) const
{
    auto ti = _timers.find(tname);
    if (ti != _timers.end()) {
        return timer_prefix_string(ti->second._type);
    }
    else {
        return std::string("unknown.");
    }
}

std::string
profiling_host::timer_prefix_string(timer_type ttype) const
{
    std::string r;

    switch (ttype) {
        case CPU_TIMER: r.assign("cpu"); break;
        case GL_TIMER:  r.assign("gl"); break;
#if SCM_ENABLE_CUDA_CL_SUPPORT          
        case CU_TIMER:  r.assign("cuda"); break;
        case CL_TIMER:  r.assign("cl"); break;
#endif
        default:        r.assign("unknown.");
    }

    return r;
}

profiling_host::timer_ptr
profiling_host::find_timer(const std::string& tname) const
{
    auto ti = _timers.find(tname);
    if (ti != _timers.end()) {
        return ti->second._timer;
    }
    else {
        return timer_ptr();
    }
}

scoped_timer::scoped_timer(profiling_host& phost, const std::string& tname)
  : _phost(phost)
  , _tname(tname)
{
    phost.cpu_start(_tname);
}

scoped_timer::scoped_timer(profiling_host& phost, const std::string& tname, const render_context_ptr& context)
  : _phost(phost)
  , _tname(tname)
{
    phost.gl_start(_tname, context);
}

#if SCM_ENABLE_CUDA_CL_SUPPORT
scoped_timer::scoped_timer(profiling_host& phost, const std::string& tname, const cu::cuda_command_stream_ptr& cu_stream)
  : _phost(phost)
  , _tname(tname)
{
    phost.cu_start(_tname, cu_stream);
}
#endif

scoped_timer::~scoped_timer()
{
    _phost.stop(_tname);
}

profiling_result::profiling_result(const profiling_host_cptr& host,
                                   const std::string&         tname,
                                         time::time_io        unit)
  : _phost(host)
  , _tname(tname)
  , _dsize(0)
  , _unit(unit)
{
}

profiling_result::profiling_result(const profiling_host_cptr& host,
                                   const std::string&         tname,
                                         scm::size_t          dsize,
                                         time::time_io        unit)
  : _phost(host)
  , _tname(tname)
  , _dsize(dsize)
  , _unit(unit)
{
}

std::string
profiling_result::unit_string() const
{
    return time::time_io::time_unit_string(_unit._t_unit);
}

std::string
profiling_result::throughput_string() const
{
    return time::time_io::throughput_unit_string(_unit._tp_unit);
}

double
profiling_result::time() const
{
    profiling_host::nanosec_type d = _phost->time(_tname);
    return time::time_io::to_time_unit(_unit._t_unit, d);
}

double
profiling_result::throughput() const
{
    profiling_host::nanosec_type d = _phost->time(_tname);
    return time::time_io::to_throughput_unit(_unit._tp_unit, d, _dsize);
}

std::ostream& operator<<(std::ostream& os, const profiling_result& pres)
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);
        os << std::fixed << std::setprecision(3);

        if (pres._phost->enabled()) {
            if (0 == pres._phost->time(pres._tname)) {
                os << "unused timer";
            }
            else {
                //os << std::setw(4) << std::left  << pres._phost->timer_prefix_string(pres._tname);// << ""
                   //<< std::setw(6) << std::right << pres.time() << pres.unit_string();
                //if (0 < pres._dsize) {
                //    os << ", "
                //       << std::setw(9) << std::right << pres.throughput() << pres.throughput_string();
                //}

                profiling_host::timer_ptr t = pres._phost->find_timer(pres._tname);

                if (dynamic_pointer_cast<cpu_accum_timer>(t)) {
                    os << std::setw(6) << std::left  << pres._phost->timer_prefix_string(pres._tname);// << ""
                    t->report(os, pres._dsize, pres._unit);
                }
                else {
                    t->detailed_report(os, pres._dsize, pres._unit);
                }
            }
        }
        else {
            os << "profiling disabled";
        }
    }
    else {
        os.setstate(std::ios_base::failbit);
    }

    return os;
}

} // namespace util
} // namespace gl
} // namespace scm
