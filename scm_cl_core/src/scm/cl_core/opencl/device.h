
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_OPENCL_DEVICE_H_INCLUDED
#define SCM_CL_CORE_OPENCL_DEVICE_H_INCLUDED

#include <set>
#include <string>

#include <boost/noncopyable.hpp>

#include <scm/core/math.h>
#include <scm/core/memory.h>

#include <scm/cl_core/opencl/opencl_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cl {

class __scm_export(cl_core) opencl_device : boost::noncopyable
{
////// types //////////////////////////////////////////////////////////////////////////////////////
protected:
    typedef std::set<std::string>                           string_set;

////// methods ////////////////////////////////////////////////////////////////////////////////////
public:
    opencl_device();
    virtual ~opencl_device();

    // device /////////////////////////////////////////////////////////////////////////////////////
public:
    const cl::platform_ptr&         cl_platform() const;
    const cl::context_ptr&          cl_context() const;
    const cl::device_ptr&           cl_device() const;

    virtual void                    print_device_informations(std::ostream& os) const;

    void                            add_include_path(const std::string& in_path);

    cl::program_ptr                 create_program(const std::string& in_source,
                                                   const std::string& in_options,
                                                   const std::string& in_source_name = "");
    cl::program_ptr                 create_program_from_file(const std::string& in_file_name,
                                                             const std::string& in_options);

    cl::kernel_ptr                  create_kernel(const cl::program_ptr& in_program,
                                                  const std::string&     in_entry_point);

    cl::command_queue_ptr           create_command_queue();


protected:
    bool                            init_opencl();

////// attributes /////////////////////////////////////////////////////////////////////////////////
protected:
    // device /////////////////////////////////////////////////////////////////////////////////////
    struct mutex_impl;
    shared_ptr<mutex_impl>          _mutex_impl;

    // opencl /////////////////////////////////////////////////////////////////////////////////////
    cl::platform_ptr                _cl_platform;
    cl::context_ptr                 _cl_context;
    cl::device_ptr                  _cl_device;

    std::string                     _default_options;
    string_set                      _default_cl_include_paths;

}; // class opencl_device

__scm_export(cl_core) std::ostream& operator<<(std::ostream& os, const opencl_device& ren_dev);

} // namespace cl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CL_CORE_OPENCL_DEVICE_H_INCLUDED
