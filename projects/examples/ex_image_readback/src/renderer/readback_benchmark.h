
#ifndef SCM_LARGE_DATA_READBACK_BENCHMARK_H_INCLUDED
#define SCM_LARGE_DATA_READBACK_BENCHMARK_H_INCLUDED

#include <string>
#include <vector>

#include <boost/thread.hpp>

#include <cuda_runtime.h>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/font/font_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/utilities/utilities_fwd.h>
#include <scm/gl_util/viewer/viewer_fwd.h>

struct cudaGraphicsResource;

namespace scm {
namespace data {

class readback_benchmark
{
protected:
    typedef time::accum_timer<time::high_res_timer> cpu_timer_type;

    class sync_event {
    public:
        sync_event(bool signaled = false) 
        : _signaled(signaled)
        , _mutex(new boost::mutex)
        , _condition(new boost::condition_variable) {
        }
        void signal() {
            _signaled = true;
            _condition->notify_all();
        }
        void reset() {
            _signaled = false;
            _condition->notify_all();
        }
        void wait() {
            boost::mutex::scoped_lock lock(*_mutex);
            while (!_signaled) {
                _condition->wait(lock);
            }
        }

    private:
        shared_ptr<boost::mutex>                _mutex;
        shared_ptr<boost::condition_variable>   _condition;

        bool                                    _signaled;
    };

public:
    readback_benchmark(const gl::render_device_ptr& device,
                       const gl::wm::context_cptr&  gl_context,
                       const gl::wm::surface_cptr&  gl_surface,
                       const math::vec2ui&          viewport_size,
                       const std::string&           model_file);
    virtual ~readback_benchmark();

    void                            draw(const gl::render_context_ptr& context);
    void                            update(const gl::render_context_ptr& context,
                                           const gl::camera&             cam);

    bool                            capture_enabled() const;
    void                            capture_enabled(bool e);

protected:
    void                            draw_model(const gl::render_context_ptr& context);

    void                            readback_thread_entry(const gl::render_device_ptr& device,
                                                          const gl::wm::context_cptr&  gl_context,
                                                          const gl::wm::surface_cptr&  gl_surface,
                                                          const math::vec2ui&          viewport_size);


protected:
    math::vec2ui                    _viewport_size;
    gl::data_format                 _color_format;

    // readback thread
    typedef std::vector<sync_event>         sync_vector;
    typedef std::vector<gl::sync_ptr>       gl_sync_vector;
    typedef std::vector<gl::texture_2d_ptr> gl_texture_2d_vector;

    int                             _readback_buffer_count;
    shared_ptr<boost::thread>       _readback_thread;
    gl::wm::context_ptr             _readback_thread_gl_context;
    bool                            _readback_thread_running;
    gl_sync_vector                  _readback_draw_end;
    sync_vector                     _readback_draw_end_valid;
    gl_sync_vector                  _readback_readback_end;
    sync_vector                     _readback_readback_end_valid;
    gl_texture_2d_vector            _readback_textures;

    bool                            _capture_enabled;

    // capture
    gl::buffer_ptr                  _copy_buffer_0;
    gl::buffer_ptr                  _copy_buffer_1;
    gl::texture_buffer_ptr          _copy_texture_color_buffer;
    gl::texture_buffer_ptr          _copy_texture_depth_buffer;
    gl::texture_2d_ptr              _copy_color_buffer;
    gl::texture_2d_ptr              _copy_depth_buffer;
    gl::texture_2d_ptr              _copy_lock_buffer;
    gl::frame_buffer_ptr            _copy_framebuffer;
    shared_ptr<uint8>               _copy_memory;
    //shared_array<uint8>             _copy_memory;


    // cuda
    cudaStream_t                     _cuda_stream;
    shared_ptr<uint8>                _cuda_copy_memory;
    size_t                           _cuda_copy_memory_size;
    shared_ptr<cudaGraphicsResource> _cuda_gfx_res;

    // framebuffer
    //gl::render_buffer_ptr           _color_buffer;
    //gl::render_buffer_ptr           _depth_buffer;
    gl::texture_2d_ptr              _color_buffer;
    gl::texture_2d_ptr              _depth_buffer;
    gl::frame_buffer_ptr            _framebuffer;

    gl::program_ptr                 _vtexture_program;

    gl::depth_stencil_state_ptr     _dstate;
    gl::blend_state_ptr             _bstate;
    gl::rasterizer_state_ptr        _rstate;
    gl::sampler_state_ptr           _sstate;

    gl::wavefront_obj_geometry_ptr  _model;

    gl::texture_output_ptr          _texture_output;

    gl::camera_uniform_block_ptr    _camera_block;

    // timing
    gl::accum_timer_query_ptr       _gpu_timer_draw;
    gl::accum_timer_query_ptr       _gpu_timer_read;
    gl::accum_timer_query_ptr       _gpu_timer_copy;
    gl::accum_timer_query_ptr       _gpu_timer_tex;
    cpu_timer_type                  _cpu_timer_frame;
    cpu_timer_type                  _cpu_timer_read;
    cpu_timer_type                  _cpu_timer_copy;
    cpu_timer_type                  _cpu_timer_tex;

    // sync
    gl::sync_ptr                    _capture_finished;
    int64                           _capture_start_frame;

    int64                           _frame_number;

    // font
    gl::text_renderer_ptr           _text_renderer;
    gl::text_ptr                    _output_text;

}; // class readback_benchmark

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_READBACK_BENCHMARK_H_INCLUDED
