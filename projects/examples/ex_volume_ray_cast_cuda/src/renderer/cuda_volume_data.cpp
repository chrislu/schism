
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "cuda_volume_data.h"

#include <exception>
#include <sstream>
#include <stdexcept>

#include <boost/bind.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <scm/core/platform/windows.h>
#include <cuda_gl_interop.h>

#include <scm/log.h>
#include <scm/core/log/logger_state.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/viewer/camera.h>

#include <scm/cl_core/cuda.h>

#include <renderer/volume_data.h>
#include <renderer/kernel/volume_ray_cast.h>

namespace scm {
namespace data {

cuda_volume_data::cuda_volume_data(const gl::render_device_ptr& device,
                                   const volume_data_ptr&       voldata)
  : _data(voldata)
{
    log::logger_format_saver out_save(out().associated_logger());
    out() << "cuda_volume_data::cuda_volume_data(): creating CUDA volume data resources..." << log::end;
    out() << log::indent;
    {
        std::stringstream   os;
        device->dump_memory_info(os);
        out() << log::info
              << "before creating volume gl image:" << log::nline
              << os.str() << log::end;
    }

    cudaError cu_err = cudaSuccess;

    cudaGraphicsResource* cu_vol_resource = 0;
    cu_err = cudaGraphicsGLRegisterImage(&cu_vol_resource, voldata->volume_raw()->object_id(),
                                                           voldata->volume_raw()->object_target(),
                                                           cudaGraphicsRegisterFlagsReadOnly);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "cuda_volume_data::cuda_volume_data() "
              << "error registering volume texture as cuda graphics resource (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("cuda_volume_data::cuda_volume_data()..."));
    }
    else {
        _volume_image.reset(cu_vol_resource, boost::bind<cudaError>(cudaGraphicsUnregisterResource, _1));
    }

    cudaGraphicsResource* cu_ca_resource = 0;
    cu_err = cudaGraphicsGLRegisterImage(&cu_ca_resource, voldata->color_alpha_map()->object_id(),
                                                          voldata->color_alpha_map()->object_target(),
                                                          cudaGraphicsRegisterFlagsReadOnly);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "cuda_volume_data::cuda_volume_data() "
              << "error registering color-alpha texture as cuda graphics resource (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("cuda_volume_data::cuda_volume_data()..."));
    }
    else {
        _color_alpha_image.reset(cu_ca_resource, boost::bind<cudaError>(cudaGraphicsUnregisterResource, _1));
    }
}

cuda_volume_data::~cuda_volume_data()
{
    _data.reset();

    _volume_image.reset();
    _color_alpha_image.reset();
}

void
cuda_volume_data::update(const gl::render_context_ptr& context)
{
    volume_uniform_data d;
    memcpy(&d, data()->volume_block().get_block(), sizeof(volume_uniform_data));

    d._m_matrix                    = math::transpose(d._m_matrix                   );
    d._m_matrix_inverse            = math::transpose(d._m_matrix_inverse           );
    d._m_matrix_inverse_transpose  = math::transpose(d._m_matrix_inverse_transpose );
    d._mv_matrix                   = math::transpose(d._mv_matrix                  );
    d._mv_matrix_inverse           = math::transpose(d._mv_matrix_inverse          );
    d._mv_matrix_inverse_transpose = math::transpose(d._mv_matrix_inverse_transpose);
    d._mvp_matrix                  = math::transpose(d._mvp_matrix                 );
    d._mvp_matrix_inverse          = math::transpose(d._mvp_matrix_inverse         );

    cudaError cu_err = cudaSuccess;

    //cu_err = cudaMemcpyAsync(_volume_uniform_buffer.get(), &d, sizeof(volume_uniform_data), cudaMemcpyHostToDevice, cuda_stream);
    //cu_err = cudaMemcpyToSymbolAsync("uniform_data", &d, sizeof(volume_uniform_data), 0, cudaMemcpyHostToDevice, context->cuda_command_stream()->stream());
    //    
    //assert(cudaSuccess == cu_err);

    upload_uniform_data(d, context->cuda_command_stream()->stream());
}

const volume_data_ptr&
cuda_volume_data::data() const
{
    return _data;
}

const shared_ptr<cudaGraphicsResource>&
cuda_volume_data::volume_image() const
{
    return _volume_image;
}

const shared_ptr<cudaGraphicsResource>&
cuda_volume_data::color_alpha_image() const
{
    return _color_alpha_image;
}

} // namespace data
} // namespace scm
