
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

#include <scm/core/platform/windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <scm/log.h>

#include <scm/cl_core/config.h>
#include <scm/cl_core/cuda.h>

namespace scm {
namespace cu {

struct cuda_device::mutex_impl
{
    boost::mutex    _mutex;
};

cuda_device::cuda_device()
  : _mutex_impl(new mutex_impl)
  , _cuda_gl_device(-1)
{
    if (!init_cuda()) {
        std::ostringstream s;
        s << "cuda_device::cuda_device(): error initializing CUDA.";
        err() << log::fatal << s.str() << log::end;
        throw std::runtime_error(s.str());
    }
}

cuda_device::~cuda_device()
{
    cudaError cu_err = cudaDeviceReset();
    if (cudaSuccess != cu_err) {
        err() << log::error
              << "cuda_device::~cuda_device() "
              << "error resetting device " << _cuda_gl_device << " (" << cudaGetErrorString(cu_err) << ")." << log::end;
    }
}

int
cuda_device::cu_device() const
{
    return _cuda_gl_device;
}

cuda_command_stream_ptr
cuda_device::create_command_stream()
{
    cuda_command_stream_ptr p;
    try {
        p.reset(new cuda_command_stream(*this));
    }
    catch (const std::exception& e) {
        err() << log::fatal
              << "cuda_device::init_cuda() "
              << "error creating cuda stream (" << e.what() << ")." << log::end;
        return cuda_command_stream_ptr();
    }

    return p;
}

void
cuda_device::print_device_informations(std::ostream& os) const
{
    assert(_cuda_gl_device > -1);
    { // protect this function from multiple thread access
        boost::mutex::scoped_lock lock(_mutex_impl->_mutex);

        using namespace std;
        
        cudaError      cu_err = cudaSuccess;
        cudaDeviceProp cu_dev_props;
        cu_err = cudaGetDeviceProperties(&cu_dev_props, _cuda_gl_device);
        if (cudaSuccess != cu_err) {
            err() << log::fatal
                  << "cuda_device::print_device_informations() "
                  << "error getting cuda device properties for dev_id " << _cuda_gl_device << " (" << cudaGetErrorString(cu_err) << ")." << log::end;
            return;
        }
        else {
            std::string cu_compute_mode;
            switch(cu_dev_props.computeMode) {
                case cudaComputeModeDefault:          cu_compute_mode.assign("cudaComputeModeDefault"); break;
                case cudaComputeModeExclusive:        cu_compute_mode.assign("cudaComputeModeExclusive"); break;
                case cudaComputeModeProhibited:       cu_compute_mode.assign("cudaComputeModeProhibited"); break;
                case cudaComputeModeExclusiveProcess: cu_compute_mode.assign("cudaComputeModeExclusiveProcess"); break;
                default:                              cu_compute_mode.assign("unknown mode"); break;
            }

            os  << "CUDA device info (dev_id " << _cuda_gl_device << "):" << std::endl
                << std::setprecision(2) << std::fixed
                << "    - name:                             " << cu_dev_props.name                                                        << std::endl 
                << "    - clock rate:                       " << static_cast<double>(cu_dev_props.clockRate) / 1000.0 << "MHz"            << std::endl /**< Clock frequency in kilohertz */
                << "    - clock rate (memory):              " << static_cast<double>(cu_dev_props.memoryClockRate) / 1000.0 << "MHz"      << std::endl /**< Peak memory clock frequency in kilohertz */
                << "    - memory bus width:                 " << cu_dev_props.memoryBusWidth << "bit"                                     << std::endl /**< Global memory bus width in bits */
                << "    - compute capability:               " << cu_dev_props.major << "." << cu_dev_props.minor                          << std::endl /**< compute capability */
                << "    - async engine count:               " << cu_dev_props.asyncEngineCount                                            << std::endl /**< Number of asynchronous engines */
                << "    - total global memory:              " << static_cast<double>(cu_dev_props.totalGlobalMem) / 1048576.0 << "GiB"    << std::endl /**< Global memory available on device in bytes */
                << "    - shared memory per block:          " << static_cast<double>(cu_dev_props.sharedMemPerBlock) / 1024.0 << "KiB"    << std::endl /**< Shared memory available per block in bytes */
                << "    - total const memory:               " << static_cast<double>(cu_dev_props.totalConstMem) / 1024.0 << "KiB"        << std::endl /**< Constant memory available on device in bytes */
                << "    - registers per block:              " << cu_dev_props.regsPerBlock                                                << std::endl /**< 32-bit registers available per block */
                << "    - L2 cache size:                    " << static_cast<double>(cu_dev_props.l2CacheSize) / 1024.0 << "KiB"          << std::endl /**< Size of L2 cache in bytes */
                << "    - warp size:                        " << cu_dev_props.warpSize << " threads"                                      << std::endl /**< Warp size in threads */
                << "    - memory pitch:                     " << cu_dev_props.memPitch << "B"                                             << std::endl /**< Maximum pitch in bytes allowed by memory copies */
                << "    - max threads per block:            " << cu_dev_props.maxThreadsPerBlock << " threads"                            << std::endl /**< Maximum number of threads per block */
                << "    - max threads per multi-processor:  " << cu_dev_props.maxThreadsPerMultiProcessor << " threads"                   << std::endl /**< Maximum resident threads per multiprocessor */
                << "    - max threads dimension:            " << "("  << cu_dev_props.maxThreadsDim[0]
                                                              << ", " << cu_dev_props.maxThreadsDim[1]
                                                              << ", " << cu_dev_props.maxThreadsDim[2] << ")"                             << std::endl  /**< Maximum size of each dimension of a block */
                << "    - max grid size:                    " << "("  << cu_dev_props.maxGridSize[0]
                                                              << ", " << cu_dev_props.maxGridSize[1]
                                                              << ", " << cu_dev_props.maxGridSize[2] << ")"                               << std::endl/**< Maximum size of each dimension of a grid */
                << "    - texture alignment:                " << cu_dev_props.textureAlignment                                            << std::endl /**< Alignment requirement for textures */
                << "    - texture pitch alignment:          " << cu_dev_props.texturePitchAlignment                                       << std::endl /**< Pitch alignment requirement for texture references bound to pitched memory */
                << "    - multiProcessorCount:              " << cu_dev_props.multiProcessorCount                                         << std::endl /**< Number of multiprocessors on device */
                << "    - kernel exec timeout enabled:      " << cu_dev_props.kernelExecTimeoutEnabled                                    << std::endl /**< Specified whether there is a run time limit on kernels */
                << "    - integrated:                       " << cu_dev_props.integrated                                                  << std::endl /**< Device is integrated as opposed to discrete */
                << "    - can map host memory:              " << cu_dev_props.canMapHostMemory                                            << std::endl /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
                << "    - compute mode:                     " << cu_compute_mode                                                          << std::endl /**< Compute mode (See ::cudaComputeMode) */
                << "    - surface alignment:                " << cu_dev_props.surfaceAlignment                                            << std::endl /**< Alignment requirements for surfaces */
                << "    - concurrent kernels:               " << cu_dev_props.concurrentKernels                                           << std::endl /**< Device can possibly execute multiple kernels concurrently */
                << "    - ECC enabled:                      " << cu_dev_props.ECCEnabled                                                  << std::endl /**< Device has ECC support enabled */
                << "    - pci bus ID:                       " << cu_dev_props.pciBusID                                                    << std::endl /**< PCI bus ID of the device */
                << "    - pci device ID:                    " << cu_dev_props.pciDeviceID                                                 << std::endl /**< PCI device ID of the device */
                << "    - pci domain ID:                    " << cu_dev_props.pciDomainID                                                 << std::endl /**< PCI domain ID of the device */
                << "    - tcc driver:                       " << cu_dev_props.tccDriver                                                   << std::endl /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
                << "    - unified addressing:               " << cu_dev_props.unifiedAddressing                                           << std::endl /**< Device shares a unified address space with the host */
                << std::endl;
        }
    }
}

bool
cuda_device::init_cuda()
{
    cudaError cu_err = cudaSuccess;

    const int max_cugl_devices = 10;
    unsigned cugl_dev_count = 0;
    int      cugl_devices[max_cugl_devices];
    cu_err = cudaGLGetDevices(&cugl_dev_count, cugl_devices, max_cugl_devices, cudaGLDeviceListCurrentFrame);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "cuda_device::init_cuda() "
              << "error acquiring cuda-gl devices (" << cudaGetErrorString(cu_err) << ")." << log::end;
        return false;
    }
    if (cugl_dev_count < 1) {
        err() << log::fatal
              << "cuda_device::init_cuda() "
              << "error acquiring cuda-gl devices (no CUDA devices returned)." << log::end;
        return false;
    }

    cu_err = cudaGLSetGLDevice(cugl_devices[0]);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "cuda_device::init_cuda() "
              << "error setting cuda-gl device " << cugl_devices[0] << " (" << cudaGetErrorString(cu_err) << ")." << log::end;
        return false;
    }
    else {
        _cuda_gl_device = cugl_devices[0];
    }

    //_main_cuda_stream = create_command_stream();
    //if (!_main_cuda_stream) {
    //    err() << log::fatal
    //          << "cuda_device::init_cuda() "
    //          << "error creating cuda command stream." << log::end;

    //    return false;
    //}

    return true;
}



std::ostream& operator<<(std::ostream& os, const cuda_device& cu_dev)
{
    cu_dev.print_device_informations(os);
    return os;
}

} // namespace cu
} // namespace scm
