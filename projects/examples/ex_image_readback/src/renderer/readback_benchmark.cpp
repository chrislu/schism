
#include "readback_benchmark.h"

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign/list_of.hpp>


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <scm/core/platform/windows.h>
#include <cuda_gl_interop.h>

#include <scm/core/time/time_types.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/frame_buffer_objects.h>
#include <scm/gl_core/query_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/sync_objects.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/surface.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>
#include <scm/gl_util/primitives/wavefront_obj.h>
#include <scm/gl_util/utilities/accum_timer_query.h>
#include <scm/gl_util/utilities/texture_output.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/viewer/camera_uniform_block.h>

//#include <scm/large_data/virtual_texture/vtexture.h>
//#include <scm/large_data/virtual_texture/vtexture_context.h>
//#include <scm/large_data/virtual_texture/vtexture_system.h>

#include <scm/core/platform/windows.h>

#define SCM_TEST_READPIXELS 1
//#undef SCM_TEST_READPIXELS

#define SCM_TEST_TEXTURE_IMAGE_STORE 1
#undef SCM_TEST_TEXTURE_IMAGE_STORE

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE
#   define SCM_TEST_TEXTURE_IMAGE_STORE_READBACK
#   undef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK

#   define SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
#   undef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED

#   define SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA
#   undef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA

#endif // SCM_TEST_TEXTURE_IMAGE_STORE

namespace {

const int SCM_FEEDBACK_BUFFER_REDUCTION = 1;

}

namespace scm {
namespace data {

readback_benchmark::readback_benchmark(const gl::render_device_ptr& device,
                                       const gl::wm::context_cptr&  gl_context,
                                       const gl::wm::surface_cptr&  gl_surface,
                                       const math::vec2ui&          viewport_size,
                                       const std::string&           model_file)
  : _viewport_size(viewport_size)
  , _color_format(gl::FORMAT_BGRA_8)
  , _capture_enabled(true)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;
    using boost::lexical_cast;

    camera_uniform_block::add_block_include_string(device);

    _camera_block.reset(new gl::camera_uniform_block(device));

    // state objects //////////////////////////////////////////////////////////////////////////////
    _bstate = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _rstate = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);
    _sstate = device->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);

    if (   !_bstate
        || !_dstate
        || !_rstate
        || !_sstate) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating state objects");
    }

    // ray casting program ////////////////////////////////////////////////////////////////////////
    _vtexture_program = device->create_program(list_of(device->create_shader_from_file(STAGE_VERTEX_SHADER,
                                                            "../../../src/renderer/shaders/vtexture_model.glslv"))
#ifdef SCM_TEST_TEXTURE_IMAGE_STORE
                                                      (device->create_shader_from_file(STAGE_FRAGMENT_SHADER,
                                                            "../../../src/renderer/shaders/vtexture_model.glslf",
                                                            shader_macro_array("SCM_TEST_TEXTURE_IMAGE_STORE", "1")
                                                                              ("SCM_FEEDBACK_BUFFER_REDUCTION", lexical_cast<std::string>(SCM_FEEDBACK_BUFFER_REDUCTION)))),
#else
                                                      (device->create_shader_from_file(STAGE_FRAGMENT_SHADER, "../../../src/renderer/shaders/vtexture_model.glslf")),
#endif // SCM_TEST_TEXTURE_IMAGE_STORE
                                               "readback_benchmark::vtexture_program");

    if (!_vtexture_program) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating vtexture_program");
    }
    
    _vtexture_program->uniform_buffer("camera_matrices", 0);

    // framebuffer ////////////////////////////////////////////////////////////////////////////////
    out() << "_viewport_size " << _viewport_size << log::end;
    //_color_buffer          = device->create_render_buffer(_viewport_size, _color_format);
    //_depth_buffer          = device->create_render_buffer(_viewport_size, FORMAT_D24_S8);
    _color_buffer          = device->create_texture_2d(_viewport_size, _color_format);
    _depth_buffer          = device->create_texture_2d(_viewport_size, FORMAT_D24_S8);

    if (   !_color_buffer
        || !_depth_buffer) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating textures");
    }

    _framebuffer           = device->create_frame_buffer();

    if (!_framebuffer) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating framebuffers");
    }

    _framebuffer->attach_color_buffer(0, _color_buffer);
    _framebuffer->attach_depth_stencil_buffer(_depth_buffer);

    // capture ////////////////////////////////////////////////////////////////////////////////////
    vec2ui capture_size     = _viewport_size / SCM_FEEDBACK_BUFFER_REDUCTION;
    size_t copy_buffer_size = capture_size.x * capture_size.y * size_of_format(_color_format);
    _copy_framebuffer       = device->create_frame_buffer();
    _copy_texture_color_buffer = device->create_texture_buffer(_color_format, USAGE_STATIC_COPY/*USAGE_STATIC_DRAW*/, copy_buffer_size);
    _copy_color_buffer      = device->create_texture_2d(capture_size, _color_format);
    _copy_depth_buffer      = device->create_texture_2d(capture_size, FORMAT_R_32UI);
    _copy_lock_buffer       = device->create_texture_2d(capture_size, FORMAT_R_32UI);
    _copy_buffer_0          = device->create_buffer(BIND_PIXEL_PACK_BUFFER, USAGE_STREAM_READ, copy_buffer_size);
    _copy_buffer_1          = _copy_buffer_0;//device->create_buffer(BIND_PIXEL_PACK_BUFFER, USAGE_STREAM_READ, copy_buffer_size);

    //SYSTEM_INFO sys_info;
    //::GetSystemInfo(&sys_info);
    //out() << "system page size: " << sys_info.dwPageSize << "B." << log::end;

    //_copy_memory.reset(static_cast<uint8*>(VirtualAlloc(0, copy_buffer_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)),
    //                                       boost::bind<BOOL>(VirtualFree, _1, 0, MEM_RELEASE));
    _copy_memory.reset(new uint8[copy_buffer_size]);

    if (   !_copy_framebuffer
        || !_copy_texture_color_buffer
        || !_copy_color_buffer
        || !_copy_lock_buffer
        || !_copy_depth_buffer
        || !_copy_buffer_0
        || !_copy_buffer_1) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating copy buffer or texture");
    }
    _copy_framebuffer->attach_color_buffer(0, _copy_color_buffer);
    _copy_framebuffer->attach_color_buffer(1, _copy_depth_buffer);
    _copy_framebuffer->attach_color_buffer(2, _copy_lock_buffer);

    // timing /////////////////////////////////////////////////////////////////////////////////////
    _gpu_timer_draw = make_shared<accum_timer_query>(device);
    _gpu_timer_read = make_shared<accum_timer_query>(device);
    _gpu_timer_copy = make_shared<accum_timer_query>(device);
    _gpu_timer_tex  = make_shared<accum_timer_query>(device);

    if (   !_gpu_timer_draw
        || !_gpu_timer_read
        || !_gpu_timer_copy
        || !_gpu_timer_tex) {
        throw std::runtime_error("readback_benchmark::readback_benchmark(): error creating timer queries");
    }

    try {
        _texture_output = make_shared<texture_output>(device);
        _model          = scm::make_shared<wavefront_obj_geometry>(device, model_file);

        // text
        font_face_ptr output_font(new font_face(device, "../../../res/fonts/Consola.ttf", 12, 0, font_face::smooth_lcd));
        _text_renderer  = make_shared<text_renderer>(device);
        _output_text    = make_shared<text>(device, output_font, font_face::style_regular, "sick, sad world...");

        mat4f   fs_projection = make_ortho_matrix(0.0f, static_cast<float>(_viewport_size.x),
                                                  0.0f, static_cast<float>(_viewport_size.y), -1.0f, 1.0f);
        _text_renderer->projection_matrix(fs_projection);

        _output_text->text_color(math::vec4f(1.0f, 1.0f, 0.0f, 1.0f));
        _output_text->text_kerning(true);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark(): ") + e.what());
    }


    _frame_number = 0;

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
    // readback thread
    _readback_buffer_count   = 2;

    try {
        _readback_thread_gl_context.reset(new gl::wm::context(gl_surface, gl_context->context_attributes(), gl_context));
    }
    catch (std::exception& e) {
        err() << "readback_benchmark::readback_benchmark() "
              << "error creating gl context resources (" << e.what() << ")" << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark(): ") + e.what());
    }

    for (int rbi = 0; rbi < _readback_buffer_count; ++rbi) {

        _readback_draw_end.push_back(gl::sync_ptr());
        _readback_draw_end_valid.push_back(sync_event());
        
        _readback_readback_end.push_back(gl::sync_ptr());
        _readback_readback_end_valid.push_back(sync_event());

        _readback_textures.push_back(device->create_texture_2d(capture_size, _color_format));
    }

    _readback_thread_running = true;
    _readback_thread.reset(new boost::thread(boost::bind(&readback_benchmark::readback_thread_entry, this, device, gl_context, gl_surface, viewport_size)));

    // get everything rolling
    for (int rbi = 0; rbi < _readback_buffer_count; ++rbi) {
        _readback_readback_end[rbi] = device->main_context()->insert_fence_sync();
        _readback_readback_end_valid[rbi].signal();
    }
#endif

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA
    cudaError cu_err = cudaSuccess;

    const int max_cugl_devices = 10;
    unsigned cugl_dev_count = 0;
    int      cugl_devices[max_cugl_devices];
    cu_err = cudaGLGetDevices(&cugl_dev_count, cugl_devices, max_cugl_devices, cudaGLDeviceListCurrentFrame);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error acquiring cuda-gl devices (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }
    if (cugl_dev_count < 1) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error acquiring cuda-gl devices (no CUDA devices returned)." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }

    cu_err = cudaGLSetGLDevice(cugl_devices[0]);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error setting cuda-gl device " << cugl_devices[0] << " (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }

    cu_err = cudaStreamCreate(&_cuda_stream);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error creating cuda stream (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }

    size_t fb_size = copy_buffer_size;
    _cuda_copy_memory_size = copy_buffer_size;
    uint8* cu_copy_data = 0;
    cu_err = cudaHostAlloc(reinterpret_cast<void**>(&cu_copy_data), _cuda_copy_memory_size, cudaHostAllocDefault);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error allocating cuda host pinned memory (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }
    else {
        _cuda_copy_memory.reset(cu_copy_data, boost::bind<cudaError>(cudaFreeHost, _1));
    }

    cudaGraphicsResource* cu_fb_resource = 0;
    cu_err = cudaGraphicsGLRegisterImage(&cu_fb_resource, _copy_color_buffer->object_id(), _copy_color_buffer->object_target(), cudaGraphicsRegisterFlagsReadOnly);
    if (cudaSuccess != cu_err) {
        err() << log::fatal
              << "readback_benchmark::readback_benchmark() "
              << "error registering color texture as cuda graphics resource (" << cudaGetErrorString(cu_err) << ")." << log::end;
        throw std::runtime_error(std::string("readback_benchmark::readback_benchmark()..."));
    }
    else {
        _cuda_gfx_res.reset(cu_fb_resource, boost::bind<cudaError>(cudaGraphicsUnregisterResource, _1));
    }

#endif // SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA

}

readback_benchmark::~readback_benchmark()
{
#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
    _readback_thread_running = false;
    //}
    //_requests_wait_condition.notify_one();
    _readback_thread->join();
    _readback_thread.reset();
#endif

    _texture_output.reset();
    _model.reset();
    _text_renderer.reset();
    _output_text.reset();

    _gpu_timer_draw.reset();
    _gpu_timer_read.reset();
    _gpu_timer_copy.reset();
    _gpu_timer_tex.reset();

    _vtexture_program.reset();

    _camera_block.reset();
    
    _copy_buffer_0.reset();
    _copy_buffer_1.reset();
    _copy_texture_color_buffer.reset();
    _copy_texture_depth_buffer.reset();
    _copy_lock_buffer.reset();
    _copy_color_buffer.reset();
    _copy_depth_buffer.reset();
    _copy_memory.reset();

    _color_buffer.reset();
    _depth_buffer.reset();
    _framebuffer.reset();

    _dstate.reset();
    _bstate.reset();
    _rstate.reset();
    _sstate.reset();
}

bool
readback_benchmark::capture_enabled() const
{
    return _capture_enabled;
}

void
readback_benchmark::capture_enabled(bool e)
{
    _capture_enabled = e;
    out() << "readback_benchmark::capture_enabled(): " << (_capture_enabled ? "true" : "false") << log::end;
}

void
readback_benchmark::draw_model(const gl::render_context_ptr& context)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    mat4f model_matrix = make_translation(vec3f(0.0, 0.0, -2.0f));//mat4f::identity();//make_scale(0.01f, 0.01f, 0.01f);

    _gpu_timer_draw->start(context); { // draw 
        context_framebuffer_guard       cfg(context);
        context_program_guard           cpg(context);
        context_state_objects_guard     csg(context);
        context_texture_units_guard     ctg(context);
        context_image_units_guard       cig(context);
        context_uniform_buffer_guard    cug(context);

        _vtexture_program->uniform("model_matrix", model_matrix);

        context->clear_color_buffer(_framebuffer, 0, vec4f(0.3f, 0.3f, 0.3f, 1.0f));
        context->clear_depth_stencil_buffer(_framebuffer);

        context->set_frame_buffer(_framebuffer);

        context->bind_uniform_buffer(_camera_block->block().block_buffer(), 0);
        context->bind_program(_vtexture_program);

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE
        //vec2i capture_size = vec2i(_viewport_size) / SCM_FEEDBACK_BUFFER_REDUCTION;

        context->clear_color_buffer(_copy_framebuffer, 0, vec4f(0.3f, 0.3f, 0.3f, 1.0f));
        context->clear_color_buffer(_copy_framebuffer, 1, vec4ui(0u));
        context->clear_color_buffer(_copy_framebuffer, 2, vec4ui(0u));

        //_vtexture_program->uniform("output_res",  capture_size);
        // we are on 4.2, so we do not need to set these
        //_vtexture_program->uniform("output_image",  0);
        //_vtexture_program->uniform("depth_image",   1);
        //_vtexture_program->uniform("lock_image",    2);
        //_vtexture_program->uniform("output_buffer", 3);

        context->bind_image(_copy_color_buffer,         FORMAT_RGBA_8, ACCESS_WRITE_ONLY, 0);
        context->bind_image(_copy_depth_buffer,         FORMAT_R_32UI, ACCESS_READ_WRITE, 1);
        context->bind_image(_copy_lock_buffer,          FORMAT_R_32UI, ACCESS_READ_WRITE, 2);
        context->bind_image(_copy_texture_color_buffer, FORMAT_RGBA_8, ACCESS_WRITE_ONLY, 3);
#endif // SCM_TEST_TEXTURE_IMAGE_STORE

        context->set_rasterizer_state(_rstate);
        context->set_depth_stencil_state(_dstate);
        context->set_blend_state(_bstate);

        //context->apply();
    //context->sync();
        _model->draw_raw(context);
    //context->sync();
    } _gpu_timer_draw->stop();
}

void
readback_benchmark::draw(const gl::render_context_ptr& context)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    ++_frame_number;
    static int current_draw_texture = 0;
    static int last_draw_texture    = 0;// not used

    _cpu_timer_frame.start(); {

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
        if (_capture_enabled) {
            // wait for the sync object to signal the end of the readback on the current texture
            // wait for the sync object
            _readback_readback_end_valid[current_draw_texture].wait();
            context->sync_client_wait(_readback_readback_end[current_draw_texture]);

            //out() << "draw: " << current_draw_texture;

            _readback_readback_end_valid[current_draw_texture].reset();
            _readback_readback_end[current_draw_texture].reset();

            _framebuffer->attach_color_buffer(0, _readback_textures[current_draw_texture]);
            draw_model(context);
            context->clear_frame_buffer_color_attachments(_framebuffer);

            _readback_draw_end[current_draw_texture] = context->insert_fence_sync();
            _readback_draw_end_valid[current_draw_texture].signal();

            current_draw_texture = (current_draw_texture + 1) % _readback_buffer_count;
            last_draw_texture    = (last_draw_texture    + 1) % _readback_buffer_count;
        }
        else {
            _framebuffer->attach_color_buffer(0, _readback_textures[current_draw_texture]);
            draw_model(context);
            context->clear_frame_buffer_color_attachments(_framebuffer);
        }

#else // SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
        draw_model(context);
#endif // SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
        //context->sync();

#ifdef SCM_TEST_READPIXELS
        { // readback
            size_t  data_size = _viewport_size.x * _viewport_size.y * size_of_format(_color_format);
            void* data = 0;

            _cpu_timer_read.start();
            _gpu_timer_read->start(context); {
                context->orphane_buffer(_copy_buffer_0);
                context->capture_color_buffer(_framebuffer, 0,
                                              texture_region(vec3ui(0), _viewport_size),
                                              _color_format,
                                              //FORMAT_BGRA_8,
                                              _copy_buffer_0);
            } _gpu_timer_read->stop();
            _cpu_timer_read.stop();
#if 1
            _cpu_timer_copy.start();
            _gpu_timer_copy->start(context); {
                if (data = context->map_buffer(_copy_buffer_1, ACCESS_READ_ONLY)) {
                    ::memcpy(_copy_memory.get(), data, data_size);
                }
                context->unmap_buffer(_copy_buffer_1);
            } _gpu_timer_copy->stop();
            _cpu_timer_copy.stop();

            _cpu_timer_tex.start();
            _gpu_timer_tex->start(context); {
                context->update_sub_texture(_copy_color_buffer, texture_region(vec3ui(0), vec3ui(_viewport_size, 1)), 0, _color_format, _copy_memory.get());
            } _gpu_timer_tex->stop();
            _cpu_timer_tex.stop();
#else
            _cpu_timer_tex.start();
            _gpu_timer_tex->start(context); {
                context->bind_unpack_buffer(_copy_buffer_0);
                context->update_sub_texture(_copy_color_buffer, texture_region(vec3ui(0), vec3ui(_viewport_size, 1)), 0, _color_format, size_t(0));
                context->bind_unpack_buffer(buffer_ptr());
            } _gpu_timer_tex->stop(context);
            _cpu_timer_tex.stop();
#endif
            std::swap(_copy_buffer_0, _copy_buffer_1);
        }
#endif // SCM_TEST_READPIXELS

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK
        {
            vec2ui capture_size = _viewport_size / SCM_FEEDBACK_BUFFER_REDUCTION;
            size_t data_size    = capture_size.x * capture_size.y * size_of_format(_color_format);
            void*  data         = 0;


            if (!_capture_finished) {
                _cpu_timer_read.start();
                _gpu_timer_read->start(context); {
                    context->orphane_buffer(_copy_buffer_0);
                    context->capture_color_buffer(_copy_framebuffer, 0,
                                                  texture_region(vec3ui(0), capture_size),
                                                  _color_format,
                                                  //FORMAT_BGRA_8,
                                                  _copy_buffer_0);
                    _capture_finished    = context->insert_fence_sync();
                    _capture_start_frame = _frame_number;
                } _gpu_timer_read->stop(context);
                _cpu_timer_read.stop();
            }
            //context->sync();

            //_cpu_timer_copy.start();
            //_gpu_timer_copy->start(context); {
            //    if (data = context->map_buffer(_copy_buffer_1, ACCESS_READ_ONLY)) {
            //        ::memcpy(_copy_memory.get(), data, data_size);
            //    }
            //    context->unmap_buffer(_copy_buffer_1);
            //} _gpu_timer_copy->stop(context);
            //_cpu_timer_copy.stop();
            if (context->sync_signal_status(_capture_finished) == SYNC_SIGNALED) {

                int64 fdelay = _frame_number - _capture_start_frame;
                //out() << "capture delay: " << fdelay << log::end;
                context->sync_server_wait(_capture_finished);
                //_gpu_timer_copy->start(context); {
#if 1
                    if (data = context->map_buffer(_copy_buffer_1, ACCESS_READ_ONLY)) {
                _cpu_timer_copy.start();
                        ::CopyMemory(_copy_memory.get(), data, data_size);
                        //::memcpy(_copy_memory.get(), data, data_size);
                _cpu_timer_copy.stop();
                    }
                    context->unmap_buffer(_copy_buffer_1);
                //} _gpu_timer_copy->stop(context);
#else
                {
                    context_unpack_buffer_guard upbg(context);
                    context->bind_unpack_buffer(_copy_buffer_0);
                _gpu_timer_copy->start(context);
                    context->update_sub_texture(_copy_color_buffer, texture_region(vec3ui(0), vec3ui(capture_size, 1)), 0, _color_format, static_cast<size_t>(0));
                _gpu_timer_copy->stop(context);
                }
#endif


                //_cpu_timer_tex.start();
                //_gpu_timer_tex->start(context); {
                //    context->update_sub_texture(_copy_color_buffer, texture_region(vec3ui(0), vec3ui(capture_size, 1)), 0, _color_format, _copy_memory.get());
                //} _gpu_timer_tex->stop(context);
                //_cpu_timer_tex.stop();

                _capture_finished.reset();
                std::swap(_copy_buffer_0, _copy_buffer_1);
            }
        }
#endif // SCM_TEST_TEXTURE_IMAGE_STORE

#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA
        {
            if (_capture_enabled) {

                const vec2ui buf_dim   = _viewport_size;// / SCM_FEEDBACK_BUFFER_REDUCTION;
                const double buf_size  = static_cast<double>(buf_dim.x * buf_dim.y * size_of_format(_color_format)) / (1024.0 * 1024.0); // MiB

                cudaEvent_t cu_start;
                cudaEvent_t cu_stop;
                cudaError   cu_err = cudaSuccess;

                cu_err = cudaEventCreate(&cu_start);
                cu_err = cudaEventCreate(&cu_stop);

                cudaArray*             cu_array   = 0;
                cudaGraphicsResource_t cu_gfx_res = _cuda_gfx_res.get();


                cu_err = cudaEventRecord(cu_start, _cuda_stream);

                cu_err = cudaGraphicsMapResources(1, &cu_gfx_res, _cuda_stream);
                cu_err = cudaGraphicsSubResourceGetMappedArray(&cu_array, cu_gfx_res, 0, 0);

                cu_err = cudaMemcpyFromArrayAsync(_cuda_copy_memory.get(), cu_array, 0, 0, _cuda_copy_memory_size, cudaMemcpyDeviceToHost, _cuda_stream);

                cu_err = cudaGraphicsUnmapResources(1, &cu_gfx_res, _cuda_stream);

                cu_err = cudaEventRecord(cu_stop, _cuda_stream);
                cu_err = cudaStreamSynchronize(_cuda_stream);

                float cu_copy_time = 0.0f;
                cu_err = cudaEventElapsedTime(&cu_copy_time, cu_start, cu_stop);

                if (_frame_number % 1000 == 0) {
                    const double copy_bw    = buf_size * 1000.0 / cu_copy_time; // MiB/s

                    std::stringstream os;
                    os << std::fixed << std::setprecision(3)
                       << "image size: " << buf_dim << ", color_format: " << format_string(_color_format) << ", size(color_format): " << size_of_format(_color_format) << "B, buffer size: " << buf_size << "MiB" << std::endl
                       << "cuda_copy:  " << cu_copy_time << "ms, " << copy_bw << "MiB/s" << std::endl;
                    out() << os.str();
                }
            }
        }
#endif // SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_CUDA


#ifdef SCM_TEST_TEXTURE_IMAGE_STORE_READBACK_THREADED
        _texture_output->draw_texture_2d(context, _readback_textures[current_draw_texture], vec2ui(0, _viewport_size.y / 2), _viewport_size / 2);
#else
        _texture_output->draw_texture_2d(context, _color_buffer, vec2ui(0, _viewport_size.y / 2), _viewport_size / 2);
#endif
        _texture_output->draw_texture_2d(context, _depth_buffer, vec2ui(0, 0),                    _viewport_size / 2);
        _texture_output->draw_texture_2d(context, _copy_color_buffer, _viewport_size / 2,         _viewport_size / 2);
        //_texture_output->draw_texture_2d(context, _copy_depth_buffer, vec2ui(_viewport_size.x / 2, 0), _viewport_size / 2);
        _texture_output->draw_texture_2d_uint(context, _copy_depth_buffer, vec4f(100.0f / 0x00EFFFFFF), vec2ui(_viewport_size.x / 2, 0), _viewport_size / 2);

        //_texture_output->draw_texture_2d(context, _color_buffer, vec2ui(0, 0), _viewport_size);
        //_texture_output->draw_texture_2d(context, _copy_color_buffer, vec2ui(0, 0), _viewport_size);

        _text_renderer->draw_shadowed(context, vec2i(5, _viewport_size.y - 15), _output_text);
    } _cpu_timer_frame.stop();

}

void
readback_benchmark::readback_thread_entry(const gl::render_device_ptr& device,
                                          const gl::wm::context_cptr&  gl_context,
                                          const gl::wm::surface_cptr&  gl_surface,
                                          const math::vec2ui&          viewport_size)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    out() << "thread startup..." << log::end;
#if 1
    _readback_thread_gl_context->make_current(gl_surface, true);
    gl::render_context_ptr  context = device->create_context();
    context->apply();

    vec2ui                      capture_size      = viewport_size / SCM_FEEDBACK_BUFFER_REDUCTION;
    size_t                      capture_data_size = capture_size.x * capture_size.y * size_of_format(_color_format);
    gl::frame_buffer_ptr        readback_fbo      = device->create_frame_buffer();
    std::vector<gl::buffer_ptr> readback_buffers;

    for (int rbi = 0; rbi < _readback_buffer_count; ++rbi) {
        readback_buffers.push_back(device->create_buffer(BIND_PIXEL_PACK_BUFFER, USAGE_STREAM_READ, capture_data_size));
    }

    // query objects not sharable, so we create our own here
    _gpu_timer_read = make_shared<accum_timer_query>(device);

    int current_rb_buffer = 0;//_readback_buffer_count / 2;
    int last_rb_buffer    = current_rb_buffer + 1;
   
    //std::stringstream os;
    //device->print_device_informations(os);
    //out() << os.str() << log::end;
    context->set_frame_buffer(readback_fbo);
    context->apply();

    while (_readback_thread_running) {
        if (_capture_enabled) {
            // wait for the sync object to signal the end of the rendering frame
            // wait for the sync object
            _cpu_timer_copy.start();
            _readback_draw_end_valid[current_rb_buffer].wait();
            context->sync_client_wait(_readback_draw_end[current_rb_buffer]);

            //out() << "read: " << current_rb_buffer;

            _readback_draw_end_valid[current_rb_buffer].reset();
            _readback_draw_end[current_rb_buffer].reset();
            _cpu_timer_copy.stop();

            // read framebuffer content to PBO
            readback_fbo->attach_color_buffer(0, _readback_textures[current_rb_buffer]);
            context->orphane_buffer(readback_buffers[current_rb_buffer]);
            //context->sync();
            _cpu_timer_read.start();
            _gpu_timer_read->start(context);
            context->capture_color_buffer(readback_fbo, 0,
                                          texture_region(vec3ui(0), capture_size),
                                          _color_format,
                                          readback_buffers[current_rb_buffer]);

            _gpu_timer_read->stop();
            //context->sync();
            _cpu_timer_read.stop();
            context->clear_frame_buffer_color_attachments(readback_fbo);

            // signal end of readback operation
            _readback_readback_end[current_rb_buffer] = context->insert_fence_sync();
            _readback_readback_end_valid[current_rb_buffer].signal();

            _cpu_timer_tex.start();
            void*  data         = 0;
            if (data = context->map_buffer(readback_buffers[last_rb_buffer], ACCESS_READ_ONLY)) {
                ::memcpy(_copy_memory.get(), data, capture_data_size);
            }
            context->unmap_buffer(readback_buffers[last_rb_buffer]);
            _cpu_timer_tex.stop();

            current_rb_buffer = (current_rb_buffer + 1) % _readback_buffer_count;
            last_rb_buffer    = (last_rb_buffer    + 1) % _readback_buffer_count;
        }
    }
#endif
    out() << "thread shutdown..." << log::end;
}

void
readback_benchmark::update(const gl::render_context_ptr& context,
                           const gl::camera&             cam)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    _camera_block->update(context, cam);

    _gpu_timer_draw->collect();
    _gpu_timer_read->collect();
    _gpu_timer_copy->collect();
    _gpu_timer_tex->collect();

    const vec2ui buf_dim   = _viewport_size / SCM_FEEDBACK_BUFFER_REDUCTION;
    const double buf_size  = static_cast<double>(buf_dim.x * buf_dim.y * size_of_format(_color_format)) / (1024.0 * 1024.0); // MiB

    static double draw_time = 0.0;

    static double read_time = 0.0;
    static double read_tp   = 0.0;
    static double copy_time = 0.0;
    static double copy_tp   = 0.0;
    static double tex_time  = 0.0;
    static double tex_tp    = 0.0;

    static double frame_time_cpu = 0.0;
    static double read_time_cpu = 0.0;
    static double read_tp_cpu   = 0.0;
    static double copy_time_cpu = 0.0;
    static double copy_tp_cpu   = 0.0;
    static double tex_time_cpu  = 0.0;
    static double tex_tp_cpu    = 0.0;

    if (time::to_milliseconds(_cpu_timer_frame.accumulated_duration()) > 500.0) {
        draw_time = time::to_milliseconds(_gpu_timer_draw->average_duration());

        read_time = time::to_milliseconds(_gpu_timer_read->average_duration());
        read_tp   = buf_size * 1000.0 / read_time; // MiB/s
        copy_time = time::to_milliseconds(_gpu_timer_copy->average_duration());
        copy_tp   = buf_size * 1000.0 / copy_time; // MiB/s
        tex_time  = time::to_milliseconds(_gpu_timer_tex->average_duration());
        tex_tp    = buf_size * 1000.0 / tex_time; // MiB/s

        frame_time_cpu = time::to_milliseconds(_cpu_timer_frame.average_duration());
        read_time_cpu = time::to_milliseconds(_cpu_timer_read.average_duration());
        read_tp_cpu   = buf_size * 1000.0 / read_time_cpu; // MiB/s
        copy_time_cpu = time::to_milliseconds(_cpu_timer_copy.average_duration());
        copy_tp_cpu   = buf_size * 1000.0 / copy_time_cpu; // MiB/s
        tex_time_cpu  = time::to_milliseconds(_cpu_timer_tex.average_duration());
        tex_tp_cpu    = buf_size * 1000.0 / tex_time_cpu; // MiB/s

        _gpu_timer_read->reset();
        _gpu_timer_copy->reset();
        _gpu_timer_tex->reset();

        _cpu_timer_frame.reset();
        _cpu_timer_read.reset();
        _cpu_timer_copy.reset();
        _cpu_timer_tex.reset();

        std::stringstream   os;

        os << std::fixed << std::setprecision(3)
           << "image size: " << buf_dim << ", color_format: " << format_string(_color_format) << ", size(color_format): " << size_of_format(_color_format) << "B, buffer size: " << buf_size << "MiB" << std::endl
           << "draw:       gpu " << draw_time << "ms" << std::endl
           << "read:       gpu " << read_time << "ms, " << read_tp << "MiB/s - cpu: " << read_time_cpu << "ms, " << read_tp_cpu << "MiB/s" << std::endl
           << "copy:       gpu " << copy_time << "ms, " << copy_tp << "MiB/s - cpu: " << copy_time_cpu << "ms, " << copy_tp_cpu << "MiB/s" << std::endl
           << "tex_update: gpu " << tex_time  << "ms, " << tex_tp  << "MiB/s - cpu: " << tex_time_cpu  << "ms, " << tex_tp_cpu  << "MiB/s" << std::endl
           << "frame_time: cpu " << frame_time_cpu << "ms" << std::endl;

        _output_text->text_string(os.str());
    }

}

} // namespace data
} // namespace scm
