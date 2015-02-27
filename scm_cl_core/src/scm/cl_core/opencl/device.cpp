
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "device.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/thread/mutex.hpp>

#include <scm/cl_core/opencl/CL/cl.hpp>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <scm/log.h>
#include <scm/core/io/tools.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/foreach.h>

#include <scm/cl_core/config.h>
#include <scm/cl_core/opencl.h>

namespace scm {
namespace cl {

struct opencl_device::mutex_impl
{
    boost::mutex    _mutex;
};

opencl_device::opencl_device()
  : _mutex_impl(new mutex_impl)
{
    if (!init_opencl()) {
        std::ostringstream s;
        s << "opencl_device::opencl_device(): error initializing OpenCL.";
        err() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }


}

opencl_device::~opencl_device()
{
}

void
opencl_device::print_device_informations(std::ostream& os) const
{
    assert(_cl_device);
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        using namespace std;

        os << "OpenCL device" << std::endl;
        { // some information
            string dev_name;
            string dev_vendor;
            string dev_driver_version;
            string dev_device_version;
            string dev_device_c_version;
            string dev_extensions;
            string dev_profile;
            unsigned dev_address_bits;

            _cl_device->getInfo(CL_DEVICE_NAME,             &dev_name);
            _cl_device->getInfo(CL_DEVICE_VENDOR,           &dev_vendor);
            _cl_device->getInfo(CL_DRIVER_VERSION,          &dev_driver_version);
            _cl_device->getInfo(CL_DEVICE_VERSION,          &dev_device_version);
            _cl_device->getInfo(CL_DEVICE_OPENCL_C_VERSION, &dev_device_c_version);
            _cl_device->getInfo(CL_DEVICE_PROFILE,          &dev_profile);
            _cl_device->getInfo(CL_DEVICE_ADDRESS_BITS,     &dev_address_bits);
            _cl_device->getInfo(CL_DEVICE_EXTENSIONS, &dev_extensions);

            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
            boost::char_separator<char> space_separator(" ");
            tokenizer                   extension_strings(dev_extensions, space_separator);

            os << "    device name:             " << dev_name << std::endl
               << "    device vendor:           " << dev_vendor << std::endl
               << "    device driver version:   " << dev_driver_version << std::endl
               << "    device device version:   " << dev_device_version << std::endl
               << "    device device c version: " << dev_device_c_version << std::endl
               << "    device profile:          " << dev_profile << std::endl
               << "    device address bits:     " << dev_address_bits <<std::endl
               << "    device extensions:       " << "(found)"  << std::endl;
            for (tokenizer::const_iterator i = extension_strings.begin(); i != extension_strings.end(); ++i) {
                os << "                           " << string(*i) << std::endl;
            }
        }
    }
}

const cl::platform_ptr&
opencl_device::cl_platform() const
{
    return _cl_platform;
}

const cl::context_ptr&
opencl_device::cl_context() const
{
    return _cl_context;
}

const cl::device_ptr&
opencl_device::cl_device() const
{
    return _cl_device;
}

//const cl::command_queue_ptr&
//opencl_device::cl_main_command_queue() const
//{
//    return _cl_main_command_queue;
//}

void
opencl_device::add_include_path(const std::string& in_path)
{
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        _default_cl_include_paths.insert(in_path);
    }
}

cl::program_ptr
opencl_device::create_program(const std::string& in_source,
                              const std::string& in_options,
                              const std::string& in_source_name)
{
    using boost::assign::list_of;

    cl_int          cl_error = CL_SUCCESS;
    cl::program_ptr new_cl_program;

    cl::Program::Sources cl_prog_source(1, std::make_pair(in_source.c_str(), in_source.length() + 1));
    new_cl_program.reset(new cl::Program(*_cl_context, cl_prog_source));
 
    std::stringstream prog_options;

    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        string_set::const_iterator  dip_it  = _default_cl_include_paths.begin();
        string_set::const_iterator  dip_end = _default_cl_include_paths.end();

        for (; dip_it != dip_end; ++dip_it) {
            prog_options << "-I" << *dip_it << " ";
        }
    }

    prog_options << in_options;

    cl_error = new_cl_program->build(list_of(*_cl_device), prog_options.str().c_str());

    if (CL_SUCCESS != cl_error) {
        err() << log::error
              << "opencl_device::create_program(): "
              << "error building OpenCL program (" << in_source_name << ")"                               << log::nline
              << "(OpenCL error: " << cl::util::cl_error_string(cl_error) << "):"                         << log::nline
              << "build status:  " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*_cl_device)  << log::nline
              << "build options: " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*_cl_device) << log::nline
              << "build log:     " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*_cl_device)
              << log::end;
        return cl::program_ptr();
    }
    else {
        std::string build_log(new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*_cl_device));
        if (build_log.size() > 10) {
            out() << log::info
                  << "opencl_device::create_program(): "
                  << "successfully built OpenCL program (" << in_source_name << ")"                           << log::nline
                  << "build status:  " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*_cl_device)  << log::nline
                  << "build options: " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*_cl_device) << log::nline
                  << "build log:     " << build_log
                  << log::end;
        }
    }

    return new_cl_program;
}

cl::program_ptr
opencl_device::create_program_from_file(const std::string& in_file_name,
                                        const std::string& in_options)
{
    namespace bfs = boost::filesystem;

    bfs::path       file_path(in_file_name);
    cl_int          cl_error = CL_SUCCESS;
    std::string     prog_source;

    if (!io::read_text_file(in_file_name, prog_source)) {
        err() << log::error
              << "opencl_device::create_program_from_file(): "
              << "error reading kernel file: " << in_file_name << log::end;
        return cl::program_ptr();
    }

    return create_program(prog_source, in_options, file_path.filename().string());
}

cl::kernel_ptr
opencl_device::create_kernel(const cl::program_ptr& in_program,
                             const std::string&     in_entry_point)
{
    cl_int          cl_error = CL_SUCCESS;
    cl::kernel_ptr  new_cl_kernel;

    if (in_program) {
        new_cl_kernel.reset(new cl::Kernel(*in_program, in_entry_point.c_str(), &cl_error));
        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "opencl_device::create_kernel(): "
                  << "error creating kernel '" << in_entry_point << "' (" << cl::util::cl_error_string(cl_error) << ")." << log::end;
            new_cl_kernel.reset();
        }
    }
    else {
        err() << log::error
              << "opencl_device::create_kernel(): "
              << "error: invalid cl program '" << in_entry_point << "'." << log::end;
    }

    return new_cl_kernel;
}

cl::command_queue_ptr
opencl_device::create_command_queue()
{
    using scm::cl::util::cl_error_string;

    cl_int                 cl_error = CL_SUCCESS;
    cl::command_queue_ptr  new_cl_cmd_queue;

    cl_command_queue_properties cmd_prop = 0;
    cmd_prop = CL_QUEUE_PROFILING_ENABLE;
#if SCM_CL_CORE_OPENCL_ENABLE_PROFILING
#endif // SCM_CL_CORE_OPENCL_ENABLE_PROFILING
    new_cl_cmd_queue.reset(new cl::CommandQueue(*cl_context(), *cl_device(), cmd_prop, &cl_error));
    if (CL_SUCCESS != cl_error) {
        err() << log::error
              << "opencl_device::create_command_queue(): "
              << "unable to create OpenCL command queue"
              << "(" << cl_error_string(cl_error) << ")." << log::end;
        new_cl_cmd_queue.reset();              
    }

    return new_cl_cmd_queue;
}

bool
opencl_device::init_opencl()
{
    using std::string;
    using std::vector;
    using scm::cl::util::cl_error_string;
    using boost::assign::list_of;

    cl_int cl_error = CL_SUCCESS;

    { // retrieve platform
        vector<cl::Platform> cl_platforms;
        cl_error = cl::Platform::get(&cl_platforms);
        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to retrieve OpenCL platforms "
                  << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        _cl_platform.reset(new cl::Platform(cl_platforms[0]));
    }

    vector<cl::Device> cl_devices;
    _cl_platform->getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);

    if (cl_devices.empty()) {
        err() << log::error
              << "opencl_device::init_opencl(): "
              << "unable to obtain GPU OpenCL devices." << log::end;
        return false;              
    }

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    cl_context_properties ctx_props[] =
    {
        CL_GL_CONTEXT_KHR,   reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
        CL_WGL_HDC_KHR,      reinterpret_cast<cl_context_properties>(wglGetCurrentDC()),
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((*_cl_platform)()),
        0
    };
#elif SCM_PLATFORM == SCM_PLATFORM_LINUX
    cl_context_properties ctx_props[] =
    {
        CL_GL_CONTEXT_KHR,    reinterpret_cast<cl_context_properties>(glXGetCurrentContext()),
        CL_GLX_DISPLAY_KHR,   reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay()),
        CL_CONTEXT_PLATFORM,  reinterpret_cast<cl_context_properties>((*_cl_platform)()),
        0
    };
#else
#error "unsupported platform"
#endif

    { // find devices that support gl sharing
        clGetGLContextInfoKHR_fn cl_get_GLContextInfoKHR = 0;

        cl_get_GLContextInfoKHR = 0;//reinterpret_cast<clGetGLContextInfoKHR_fn>(clGetExtensionFunctionAddress("clGetGLContextInfoKHR"));

        if (cl_get_GLContextInfoKHR == 0) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to obtain clGetGLContextInfoKHR extension." << log::end;
            return false;              
        }

#if 0
        vector<cl_device_id>    cl_gl_devices(cl_devices.size());
        scm::size_t             ret_size = 0;
        cl_error = cl_get_GLContextInfoKHR(ctx_props,
                                           CL_DEVICES_FOR_GL_CONTEXT_KHR, // CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, // 
                                           cl_gl_devices.size() * sizeof(cl_device_id),
                                           &(cl_gl_devices.front()),
                                           &ret_size);
        scm::size_t comp_gl_dev = ret_size / sizeof(cl_device_id);
        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to create OpenCL command queue"
                  << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        if (comp_gl_dev < 1) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to retrieve device for assiciation with current OpenGL context." << log::end;
            return false;              
        }

        vector<cl::Device> selected_cl_device(1);
        selected_cl_device[0] = cl::Device(cl_gl_devices[0]); // we just select the first device
#else
        cl_device_id            cl_gl_device;
        scm::size_t             ret_size = 0;
        cl_error = cl_get_GLContextInfoKHR(ctx_props,
                                           CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, // CL_DEVICES_FOR_GL_CONTEXT_KHR, // 
                                           sizeof(cl_device_id),
                                           &cl_gl_device,
                                           &ret_size);
        scm::size_t comp_gl_dev = ret_size / sizeof(cl_device_id);
        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to create OpenCL command queue"
                  << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        if (comp_gl_dev < 1) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to retrieve device for assiciation with current OpenGL context." << log::end;
            return false;              
        }

        vector<cl::Device> selected_cl_device(1);
        selected_cl_device[0] = cl_gl_device; // we just select the first device
#endif
        _cl_context.reset(new cl::Context(selected_cl_device, ctx_props, 0, 0, &cl_error));

        if (CL_SUCCESS != cl_error) {
            err() << log::error
                  << "opencl_device::init_opencl(): "
                  << "unable to create OpenCL context"
                  << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }

        _cl_device.reset(new cl::Device(selected_cl_device[0]));
    }

    //{ // create main command queue
    //    _cl_main_command_queue = create_command_queue();

    //    if (!_cl_main_command_queue) {
    //            err() << log::error
    //                  << "opencl_device::init_opencl(): "
    //                  << "unable to create main OpenCl command queue." << log::end;
    //        return false;
    //    }
    //}

    return true;
}



std::ostream& operator<<(std::ostream& os, const opencl_device& cl_dev)
{
    cl_dev.print_device_informations(os);
    return os;
}

} // namespace cl
} // namespace scm
