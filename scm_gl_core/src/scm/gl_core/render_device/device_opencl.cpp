
#include "device.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/assign/list_of.hpp>

#include <CL/cl.hpp>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <scm/core/io/tools.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/config_cl.h>
#include <scm/gl_core/log.h>
#include <scm/gl_core/opencl_interop.h>

namespace scm {
namespace gl {

const cl::platform_ptr&
render_device::cl_platform() const
{
    return _cl_platform;
}

const cl::context_ptr&
render_device::cl_context() const
{
    return _cl_context;
}

const cl::device_ptr&
render_device::cl_device() const
{
    return _cl_device;
}

void
render_device::add_cl_include_path(const std::string& in_path)
{
    _default_cl_include_paths.insert(in_path);
}

cl::program_ptr
render_device::create_cl_program(const std::string& in_source,
                                 const std::string& in_options,
                                 const std::string& in_source_name)
{
    using boost::assign::list_of;

    cl_int          cl_error = CL_SUCCESS;
    cl::program_ptr new_cl_program;

#if SCM_GL_CORE_OPENCL_ENABLE_INTEROP
    cl::Program::Sources cl_prog_source(1, std::make_pair(in_source.c_str(), in_source.length() + 1));
    new_cl_program.reset(new cl::Program(*_cl_context, cl_prog_source));
 
    std::stringstream prog_options;

    string_set::const_iterator  dip_it  = _default_cl_include_paths.begin();
    string_set::const_iterator  dip_end = _default_cl_include_paths.end();

    for (; dip_it != dip_end; ++dip_it) {
        prog_options << "-I" << *dip_it << " ";
    }
    prog_options << in_options;

    cl_error = new_cl_program->build(list_of(*_cl_device), prog_options.str().c_str());

    if (CL_SUCCESS != cl_error) {
        glerr() << log::error
                << "render_device::create_cl_program(): "
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
            glout() << log::info
                    << "render_device::create_cl_program(): "
                    << "successfully built OpenCL program (" << in_source_name << ")"                           << log::nline
                    << "build status:  " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*_cl_device)  << log::nline
                    << "build options: " << new_cl_program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*_cl_device) << log::nline
                    << "build log:     " << build_log
                    << log::end;
        }
    }
#endif // SCM_GL_CORE_OPENCL_ENABLE_INTEROP

    return new_cl_program;
}

cl::program_ptr
render_device::create_cl_program_from_file(const std::string& in_file_name,
                                           const std::string& in_options)
{
    namespace bfs = boost::filesystem;

    bfs::path       file_path(in_file_name);
    cl_int          cl_error = CL_SUCCESS;
    std::string     prog_source;

    if (!io::read_text_file(in_file_name, prog_source)) {
        glerr() << log::error
                << "render_device::create_cl_program_from_file(): "
                << "error reading kernel file: " << in_file_name << log::end;
        return cl::program_ptr();
    }

    return create_cl_program(prog_source, in_options, file_path.filename().string());
}

cl::kernel_ptr
render_device::create_cl_kernel(const cl::program_ptr& in_program,
                                const std::string&     in_entry_point)
{
    cl_int          cl_error = CL_SUCCESS;
    cl::kernel_ptr  new_cl_kernel;

#if SCM_GL_CORE_OPENCL_ENABLE_INTEROP
    if (in_program) {
        new_cl_kernel.reset(new cl::Kernel(*in_program, in_entry_point.c_str(), &cl_error));
        if (CL_SUCCESS != cl_error) {
            glerr() << log::error
                    << "render_device::create_cl_kernel(): "
                    << "error creating kernel '" << in_entry_point << "' (" << cl::util::cl_error_string(cl_error) << ")." << log::end;
        }
    }
    else {
        glerr() << log::error
                << "render_device::create_cl_kernel(): "
                << "error: invalid cl program '" << in_entry_point << "'." << log::end;
    }
#endif // SCM_GL_CORE_OPENCL_ENABLE_INTEROP

    return new_cl_kernel;
}

bool
render_device::init_opencl()
{
#if SCM_GL_CORE_OPENCL_ENABLE_INTEROP
    using std::string;
    using std::vector;
    using scm::cl::util::cl_error_string;
    using boost::assign::list_of;

    cl_int cl_error = CL_SUCCESS;

    { // retrieve platform
        vector<cl::Platform> cl_platforms;
        cl_error = cl::Platform::get(&cl_platforms);
        if (CL_SUCCESS != cl_error) {
            glerr() << log::error
                    << "render_device::init_opencl(): "
                    << "unable to retrieve OpenCL platforms "
                    << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        _cl_platform.reset(new cl::Platform(cl_platforms[0]));
    }

    vector<cl::Device> cl_devices;
    _cl_platform->getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);

    if (cl_devices.empty()) {
        glerr() << log::error
                << "render_device::init_opencl(): "
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

        cl_get_GLContextInfoKHR = reinterpret_cast<clGetGLContextInfoKHR_fn>(clGetExtensionFunctionAddress("clGetGLContextInfoKHR"));

        if (cl_get_GLContextInfoKHR == 0) {
            glerr() << log::error
                    << "render_device::init_opencl(): "
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
            glerr() << log::error
                    << "render_device::init_opencl(): "
                    << "unable to create OpenCL command queue"
                    << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        if (comp_gl_dev < 1) {
            glerr() << log::error
                    << "render_device::init_opencl(): "
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
            glerr() << log::error
                    << "render_device::init_opencl(): "
                    << "unable to create OpenCL command queue"
                    << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }
        if (comp_gl_dev < 1) {
            glerr() << log::error
                    << "render_device::init_opencl(): "
                    << "unable to retrieve device for assiciation with current OpenGL context." << log::end;
            return false;              
        }

        vector<cl::Device> selected_cl_device(1);
        selected_cl_device[0] = cl_gl_device; // we just select the first device
#endif
        _cl_context.reset(new cl::Context(selected_cl_device, ctx_props, 0, 0, &cl_error));

        if (CL_SUCCESS != cl_error) {
            glerr() << log::error
                    << "render_device::init_opencl(): "
                    << "unable to create OpenCL context"
                    << "(" << cl_error_string(cl_error) << ")." << log::end;
            return false;              
        }

        _cl_device.reset(new cl::Device(selected_cl_device[0]));
    }

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

        std::stringstream os;

        os << "render_device::init_opencl(): "
           << "created OpenCL device: " << std::endl
           << "    device name:             " << dev_name << std::endl
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

        glout() << log::info << os.str();
    }

#endif // SCM_GL_CORE_OPENCL_ENABLE_INTEROP

    return true;
}

} // namespace gl
} // namespace scm

