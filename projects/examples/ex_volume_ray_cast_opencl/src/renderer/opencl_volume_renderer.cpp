
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "opencl_volume_renderer.h"

#include <exception>
#include <string>
#include <sstream>
#include <stdexcept>

#include <CL/cl.hpp>

#include <scm/log.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>

#include <scm/cl_core/opencl.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/utilities/texture_output.h>

#include <renderer/opencl_volume_data.h>
#include <renderer/volume_uniform_data.h>

namespace {

const std::string ocl_renderer_dir = "../../../src/renderer/";

} // namespace

namespace scm {
namespace data {

opencl_volume_renderer::opencl_volume_renderer(const gl::render_device_ptr& device,
                                               const cl::opencl_device_ptr& cl_device,
                                               const math::vec2ui&          viewport_size)
  : _viewport_size(viewport_size)
  , _ray_cast_kernel_wg_size(0u)
{
    using namespace scm::gl;
    using namespace scm::math;
    using scm::cl::util::cl_error_string;

    try {
        data_format         ifmt  = FORMAT_RGBA_8;
        //size_t              isize = static_cast<size_t>(viewport_size.x) * viewport_size.y * size_of_format(ifmt);
        //shared_array<uint8> idata(new uint8[isize]);
        //memset(idata.get(), 255, isize);
        //std::vector<void*>  iinit;
        //iinit.push_back(idata.get());
        _output_texture = device->create_texture_2d(viewport_size, ifmt);//, 1, 1, 1, ifmt, iinit);
        if (!_output_texture) {
            std::stringstream os;
            os << "opencl_volume_renderer::opencl_volume_renderer(): error creating output texture." << std::endl;
            throw std::runtime_error(os.str());
        }

        cl_int cl_error = CL_SUCCESS;
        _output_cl_image.reset(new cl::Image2DGL(*cl_device->cl_context(), CL_MEM_WRITE_ONLY, // CL_MEM_READ_WRITE,
                                                 _output_texture->object_target(), 0,
                                                 _output_texture->object_id(), &cl_error));
        if (CL_SUCCESS != cl_error) {
            std::stringstream os;
            os << "opencl_volume_renderer::opencl_volume_renderer(): "
               << "error creating cl image from _output_texture (" << cl_error_string(cl_error) << ")." << std::endl;
            throw std::runtime_error(os.str());
        }
        _shared_gl_objects.clear();
        _shared_gl_objects.push_back(*_output_cl_image);

        if (!reload_kernels(device, cl_device)) {
            std::stringstream os;
            os << "opencl_volume_renderer::opencl_volume_renderer(): "
               << "error creating volume ray cast kernel." << std::endl;
            throw std::runtime_error(os.str());
        }

        _texture_presenter.reset(new gl::texture_output(device));
    }
    catch (...) {
        cleanup();
        throw;
    }
}

opencl_volume_renderer::~opencl_volume_renderer()
{
    cleanup();
}

void
opencl_volume_renderer::cleanup()
{
    _texture_presenter.reset();
    _ray_cast_kernel.reset();
    _output_cl_image.reset();
    _output_texture.reset();
}

void
opencl_volume_renderer::draw(const gl::render_context_ptr& context,
                             const cl::command_queue_ptr&  cl_queue,
                             const opencl_volume_data_ptr& vdata)
{
    using namespace scm::math;
    using scm::cl::util::cl_error_string;

    vec2ui vsize = _viewport_size;
    vec2ui lsize = vec2ui(8, 24);//_ray_cast_kernel_wg_size;
    vec2ui gsize;

    gsize.x = vsize.x % lsize.x == 0 ? vsize.x : (vsize.x / lsize.x + 1) * lsize.x;
    gsize.y = vsize.y % lsize.y == 0 ? vsize.y : (vsize.y / lsize.y + 1) * lsize.y;

    cl::NDRange global_range(gsize.x, gsize.y);
    cl::NDRange local_range(lsize.x, lsize.y);
    cl_int      cl_error = CL_SUCCESS;

    //out() << _viewport_size << log::nline
    //      << lsize << log::nline
    //      << gsize << log::end;

    std::vector<cl::Memory> acq;

    acq.push_back(*_output_cl_image);
    acq.push_back(*vdata->volume_image());
    acq.push_back(*vdata->color_alpha_image());

    int arg_count = 0;
    cl_error = _ray_cast_kernel->setArg(arg_count++, *_output_cl_image);                  assert(!cl_error_string(cl_error).empty());
    cl_error = _ray_cast_kernel->setArg(arg_count++, *vdata->volume_image());             assert(!cl_error_string(cl_error).empty());
    cl_error = _ray_cast_kernel->setArg(arg_count++, *vdata->color_alpha_image());        assert(!cl_error_string(cl_error).empty());
    cl_error = _ray_cast_kernel->setArg(arg_count++, *vdata->volume_uniform_buffer());    assert(!cl_error_string(cl_error).empty());

#if 1
    //context->sync();
    //context->cl_command_queue()->finish();

    //_acquire_timer.start();
    cl_error = cl_queue->enqueueAcquireGLObjects(&acq, 0, _cl_acquire_timer.event());                                               assert(!cl_error_string(cl_error).empty());
    //context->cl_command_queue()->finish();
    //_acquire_timer.stop();

    //_kernel_timer.start();
    cl_error = cl_queue->enqueueNDRangeKernel(*_ray_cast_kernel, ::cl::NullRange, global_range, local_range, 0, _cl_kernel_timer.event());  assert(!cl_error_string(cl_error).empty());
    //context->cl_command_queue()->finish();
    //_kernel_timer.stop();

    //_release_timer.start();
    cl_error = cl_queue->enqueueReleaseGLObjects(&acq, 0, _cl_release_timer.event());                                               assert(!cl_error_string(cl_error).empty());
    //context->cl_command_queue()->finish();
    //_release_timer.stop();


    _cl_acquire_timer.collect();
    _cl_release_timer.collect();
    _cl_kernel_timer.collect();

    //out() << time::to_milliseconds(_cl_acquire_timer.accumulated_duration());
    //out() << time::to_milliseconds(_cl_release_timer.accumulated_duration());
    //out() << time::to_milliseconds(_cl_kernel_timer.accumulated_duration());

    if (time::to_milliseconds(_cl_kernel_timer.accumulated_duration()) > 300.0) {
        double acq_time = time::to_milliseconds(_cl_acquire_timer.average_duration());
        double rel_time = time::to_milliseconds(_cl_release_timer.average_duration());
        double krn_time = time::to_milliseconds(_cl_kernel_timer.average_duration());

        _cl_acquire_timer.reset();
        _cl_release_timer.reset();
        _cl_kernel_timer.reset();
        std::stringstream   os;
        os << std::fixed << std::setprecision(3)
           << "acquire_time: "   << acq_time << "ms"<< std::endl
           << "release_time: "   << rel_time << "ms"<< std::endl
           << "kernel_time : "   << krn_time << "ms"<< std::endl;
        out() << log::info << os.str() << log::end;
    }
    //if (time::to_milliseconds(_kernel_timer.accumulated_duration()) > 300.0) {
    //    double acq_time = time::to_milliseconds(_acquire_timer.average_duration());
    //    double rel_time = time::to_milliseconds(_release_timer.average_duration());
    //    double krn_time = time::to_milliseconds(_kernel_timer.average_duration());

    //    _acquire_timer.reset();
    //    _release_timer.reset();
    //    _kernel_timer.reset();
    //    std::stringstream   os;
    //    os << std::fixed << std::setprecision(3)
    //       << "acquire_time: "   << acq_time << "ms"<< std::endl
    //       << "release_time: "   << rel_time << "ms"<< std::endl
    //       << "kernel_time : "   << krn_time << "ms"<< std::endl;
    //    out() << log::info << os.str() << log::end;
    //}
#else
    cl_error = context->cl_command_queue()->enqueueAcquireGLObjects(&acq);                                               assert(!cl_error_string(cl_error).empty());
    cl_error = context->cl_command_queue()->enqueueNDRangeKernel(*_ray_cast_kernel, ::cl::NullRange, global_range, local_range, 0, 0);  assert(!cl_error_string(cl_error).empty());
    cl_error = context->cl_command_queue()->enqueueReleaseGLObjects(&acq);                                               assert(!cl_error_string(cl_error).empty());
#endif
}

void
opencl_volume_renderer::present(const gl::render_context_ptr& context) const
{
    using namespace scm::math;

    //context->cl_command_queue()->finish();

    _texture_presenter->draw_texture_2d(context, _output_texture, vec2ui(0u), _viewport_size);
}

bool
opencl_volume_renderer::reload_kernels(const gl::render_device_ptr& device,
                                       const cl::opencl_device_ptr& cl_device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using scm::cl::util::cl_error_string;

    scm::out() << log::info
               << "opencl_volume_renderer::reload_kernels(): "
               << "reloading kernel strings." << log::end;


    cl_device->add_include_path(ocl_renderer_dir);
    _ray_cast_kernel = cl_device->create_kernel(cl_device->create_program_from_file(ocl_renderer_dir + "kernel/volume_ray_cast.cl",
                                                                                    "-cl-fast-relaxed-math -cl-unsafe-math-optimizations -cl-mad-enable -cl-nv-verbose -cl-nv-opt-level=3"),//
                                                "main_vrc");

    if (   !_ray_cast_kernel) {
        err() << log::error
                << "opencl_volume_renderer::reload_kernels(): "
                << "error creating ray cast kernel." << log::end;
        return false;
    }

    size_t wg_size;
    _ray_cast_kernel->getWorkGroupInfo(*cl_device->cl_device(), CL_KERNEL_WORK_GROUP_SIZE, &wg_size);

    _ray_cast_kernel_wg_size.y = 32;
    _ray_cast_kernel_wg_size.x = static_cast<unsigned>(wg_size) / 32;

    out() << log::info << "opencl_volume_renderer::reload_kernels(): " << "_ray_cast_kernel_wg_size = " << _ray_cast_kernel_wg_size << log::end;

    cl_int cl_error = CL_SUCCESS;
    //cl_error = _ray_cast_kernel->setArg(0, *_output_cl_image);      assert(!cl_error_string(cl_error).empty());

    if (cl_error != CL_SUCCESS) {
        return false;
    }

    return true;
}

} // namespace data
} // namespace scm
