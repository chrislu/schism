
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "cuda_volume_renderer.h"

#include <exception>
#include <string>
#include <sstream>
#include <stdexcept>

#include <boost/bind.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <scm/core/platform/windows.h>
#include <cuda_gl_interop.h>

#include <scm/log.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/utilities/texture_output.h>

#include <scm/cl_core/cuda.h>

#include <renderer/cuda_volume_data.h>
#include <renderer/volume_uniform_data.h>
#include <renderer/kernel/volume_ray_cast.h>

namespace {

const std::string ocl_renderer_dir = "../../../src/renderer/";

} // namespace

namespace scm {
namespace data {

cuda_volume_renderer::cuda_volume_renderer(const gl::render_device_ptr& device,
                                           const math::vec2ui&          viewport_size)
  : _viewport_size(viewport_size)
{
    using namespace scm::gl;
    using namespace scm::math;

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
            os << "cuda_volume_renderer::cuda_volume_renderer(): error creating output texture." << std::endl;
            throw std::runtime_error(os.str());
        }

        _texture_presenter.reset(new gl::texture_output(device));

        cudaError   cu_err = cudaSuccess;

        cudaGraphicsResource* cu_oi_resource = 0;
        cu_err = cudaGraphicsGLRegisterImage(&cu_oi_resource, _output_texture->object_id(),
                                                              _output_texture->object_target(),
                                                              cudaGraphicsRegisterFlagsSurfaceLoadStore);//cudaGraphicsRegisterFlagsWriteDiscard);
        if (cudaSuccess != cu_err) {
            err() << log::fatal
                  << "cuda_volume_renderer::cuda_volume_renderer() "
                  << "error registering output texture as cuda graphics resource (" << cudaGetErrorString(cu_err) << ")." << log::end;
            throw std::runtime_error(std::string("cuda_volume_renderer::cuda_volume_renderer()..."));
        }
        else {
            _output_image.reset(cu_oi_resource, boost::bind<cudaError>(cudaGraphicsUnregisterResource, _1));
        }

#if 0
        cudaFuncAttributes  cu_krnl_attr;
        cu_err = cudaFuncGetAttributes(&cu_krnl_attr, "main_vrc");
        if (cudaSuccess != cu_err) {
            err() << log::error
                  << "cuda_volume_renderer::cuda_volume_renderer() "
                  << "error acquiring kernel attributes for kernel 'main_vrc' (" << cudaGetErrorString(cu_err) << ")." << log::end;
            //throw std::runtime_error(std::string("cuda_volume_renderer::cuda_volume_renderer()..."));
        }
        else {
            out() << log::info
                  << "cuda_volume_renderer::cuda_volume_renderer(): kernel 'main_vrc' attributes:" << log::nline
                  << "    - shared size:           " << cu_krnl_attr.sharedSizeBytes << "B"     << log::nline
                  << "    - constant size:         " << cu_krnl_attr.constSizeBytes << "B"      << log::nline
                  << "    - local size:            " << cu_krnl_attr.localSizeBytes << "B"      << log::nline
                  << "    - max threads per block: " << cu_krnl_attr.maxThreadsPerBlock         << log::nline
                  << "    - num register:          " << cu_krnl_attr.numRegs                    << log::nline
                  << "    - ptx version:           " << cu_krnl_attr.ptxVersion                 << log::nline
                  << "    - binary version:        " << cu_krnl_attr.binaryVersion              << log::nline
                  << log::end;
        }
#endif
    }
    catch (...) {
        cleanup();
        throw;
    }



}

cuda_volume_renderer::~cuda_volume_renderer()
{
    cleanup();
}

void
cuda_volume_renderer::cleanup()
{
    _output_image.reset();
    _texture_presenter.reset();
    _output_texture.reset();
}

void
cuda_volume_renderer::draw(const gl::render_context_ptr& context,
                           const cuda_volume_data_ptr&   vdata,
                           int                           render_mode,
                           int                           sample_count,
                           bool                          use_supersampling)
{
    using namespace scm::math;

    cudaError    cu_err    = cudaSuccess;
    cudaStream_t cu_stream = context->cuda_command_stream()->stream();

    // map opengl resources
    cudaEvent_t cu_start;
    cudaEvent_t cu_stop;

    cu_err = cudaEventCreate(&cu_start);
    cu_err = cudaEventCreate(&cu_stop);


    cudaGraphicsResource_t cu_gfx_res[] = { _output_image.get(),
                                            vdata->volume_image().get(),
                                            vdata->color_alpha_image().get() };
    int                    cu_gfx_res_count = 3;

    cu_err = cudaEventRecord(cu_start, cu_stream);
    cu_err = cudaGraphicsMapResources(cu_gfx_res_count, cu_gfx_res, cu_stream);

    cuda::startup_ray_cast_kernel(_viewport_size.x, _viewport_size.y,
                                  _output_image.get(),
                                  vdata->volume_image().get(),
                                  vdata->color_alpha_image().get(),
                                  render_mode,
                                  sample_count,
                                  use_supersampling,
                                  cu_stream);

    cu_err = cudaGraphicsUnmapResources(cu_gfx_res_count, cu_gfx_res, cu_stream);
    cu_err = cudaEventRecord(cu_stop, cu_stream);
    cu_err = cudaStreamSynchronize(cu_stream);

    float cu_copy_time = 0.0f;
    cu_err = cudaEventElapsedTime(&cu_copy_time, cu_start, cu_stop);

    //if (_frame_number % 1000 == 0) {
    //    const double copy_bw    = buf_size * 1000.0 / cu_copy_time; // MiB/s

    //    std::stringstream os;
    //    os << std::fixed << std::setprecision(3)
    //        << "image size: " << buf_dim << ", color_format: " << format_string(_color_format) << ", size(color_format): " << size_of_format(_color_format) << "B, buffer size: " << buf_size << "MiB" << std::endl
    //        << "cuda_copy:  " << cu_copy_time << "ms, " << copy_bw << "MiB/s" << std::endl;
    //    out() << os.str();
    //}

}

void
cuda_volume_renderer::present(const gl::render_context_ptr& context) const
{
    using namespace scm::math;

    //context->cl_command_queue()->finish();

    _texture_presenter->draw_texture_2d(context, _output_texture, vec2ui(0u), _viewport_size);
}

} // namespace data
} // namespace scm
