
#ifndef SCM_CL_CORE_CUDA_FWD_H_INCLUDED
#define SCM_CL_CORE_CUDA_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace cu {

class cuda_device;

typedef scm::shared_ptr<cuda_device>        cuda_device_ptr;
typedef scm::shared_ptr<cuda_device const>  cuda_device_cptr;

class cuda_command_stream;

typedef scm::shared_ptr<cuda_command_stream>        cuda_command_stream_ptr;
typedef scm::shared_ptr<cuda_command_stream const>  cuda_command_stream_cptr;

} // namespace cu
} // namespace scm

#endif // SCM_CL_CORE_CUDA_FWD_H_INCLUDED