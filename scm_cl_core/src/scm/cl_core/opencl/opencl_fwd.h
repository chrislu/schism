
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CL_CORE_OPENCL_FWD_H_INCLUDED
#define SCM_CL_CORE_OPENCL_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace cl {

class Platform;
class Device;
class Context;
class Event;
class UserEvent;
class Memory;
class Buffer;
class BufferGL;
class BufferRenderGL;
class Image;
class Image2D;
class Image2DGL;
class Image3D;
class Image3DGL;
class Sampler;
class Program;
class CommandQueue;
class Kernel;
class NDRange;

} // namespace cl

namespace scm {
namespace cl {
namespace util {

class accum_timer;

typedef scm::shared_ptr<accum_timer>        accum_timer_ptr;
typedef scm::shared_ptr<accum_timer const>  accum_timer_cptr;

} // namespace util

class opencl_device;

typedef scm::shared_ptr<opencl_device>        opencl_device_ptr;
typedef scm::shared_ptr<opencl_device const>  opencl_device_cptr;

using ::cl::Platform;
using ::cl::Device;
using ::cl::Context;
using ::cl::Event;
using ::cl::UserEvent;
using ::cl::Memory;
using ::cl::Buffer;
using ::cl::BufferGL;
using ::cl::BufferRenderGL;
using ::cl::Image;
using ::cl::Image2D;
using ::cl::Image2DGL;
using ::cl::Image3D;
using ::cl::Image3DGL;
using ::cl::Sampler;
using ::cl::Program;
using ::cl::CommandQueue;
using ::cl::Kernel;
using ::cl::NDRange;

typedef scm::shared_ptr<Platform>        platform_ptr;
typedef scm::shared_ptr<Platform const>  platform_cptr;

typedef scm::shared_ptr<Device>        device_ptr;
typedef scm::shared_ptr<Device const>  device_cptr;

typedef scm::shared_ptr<Context>        context_ptr;
typedef scm::shared_ptr<Context const>  context_cptr;

typedef scm::shared_ptr<Event>        event_ptr;
typedef scm::shared_ptr<Event const>  event_cptr;

typedef scm::shared_ptr<UserEvent>        user_event_ptr;
typedef scm::shared_ptr<UserEvent const>  user_event_cptr;

typedef scm::shared_ptr<Memory>        memory_ptr;
typedef scm::shared_ptr<Memory const>  memory_cptr;

typedef scm::shared_ptr<Buffer>        buffer_ptr;
typedef scm::shared_ptr<Buffer const>  buffer_cptr;

typedef scm::shared_ptr<BufferGL>        guffer_gl_ptr;
typedef scm::shared_ptr<BufferGL const>  guffer_gl_cptr;

typedef scm::shared_ptr<BufferRenderGL>        buffer_render_gl_ptr;
typedef scm::shared_ptr<BufferRenderGL const>  buffer_render_gl_cptr;

typedef scm::shared_ptr<Image>        image_ptr;
typedef scm::shared_ptr<Image const>  image_cptr;

typedef scm::shared_ptr<Image2D>        image_2d_ptr;
typedef scm::shared_ptr<Image2D const>  image_2d_cptr;

typedef scm::shared_ptr<Image2DGL>        image_2d_gl_ptr;
typedef scm::shared_ptr<Image2DGL const>  image_2d_gl_cptr;

typedef scm::shared_ptr<Image3D>        image_3d_ptr;
typedef scm::shared_ptr<Image3D const>  image_3d_cptr;

typedef scm::shared_ptr<Image3DGL>        image_3d_gl_ptr;
typedef scm::shared_ptr<Image3DGL const>  image_3d_gl_cptr;

typedef scm::shared_ptr<Sampler>        sampler_ptr;
typedef scm::shared_ptr<Sampler const>  sampler_cptr;

typedef scm::shared_ptr<Program>        program_ptr;
typedef scm::shared_ptr<Program const>  program_cptr;

typedef scm::shared_ptr<CommandQueue>        command_queue_ptr;
typedef scm::shared_ptr<CommandQueue const>  command_queue_cptr;

typedef scm::shared_ptr<Kernel>        kernel_ptr;
typedef scm::shared_ptr<Kernel const>  kernel_cptr;

typedef scm::shared_ptr<NDRange>        nd_range_ptr;
typedef scm::shared_ptr<NDRange const>  nd_range_cptr;

} // namespace cl
} // namespace scm

#endif // SCM_CL_CORE_OPENCL_FWD_H_INCLUDED
