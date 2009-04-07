
#ifndef SCM_GL_SCMGL_SCMGL_H_INCLUDED
#define SCM_GL_SCMGL_SCMGL_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/formats.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class device;

typedef scm::shared_ptr<void>   handle;

enum device_type
{
    SCMGL_DEVICE_NULL       = 0x00,

    SCMGL_DEVICE_OPENGL,
    SCMGL_DEVICE_DIREC3D
};

struct device_initializer
{
    device_type             _type;
    std::vector<unsigned>   _request_feature_levels;
    // adapter_identifier   _output_adapter;

    device_initializer();
}; // struct device_initializer

struct device_context_config
{
    enum context_type {
        CONTEXT_WINDOWED        = 0x01,
        CONTEXT_FULLSCREEN,
        CONTEXT_OFFSCREEN,
        CONTEXT_DEFERRED
    };

    // output window
    handle              _output_window;
    scm::math::vec2ui   _output_dimensions;

    // back buffer format
    data_format         _color_buffer_format;           // color buffer mode
    data_format         _depth_stencil_buffer_format;   // depth, stencil buffer format
    unsigned            _color_buffer_count;            // number of color buffers in swap chain
    unsigned            _sample_count;                  // multi sample mode (1x default,...)

    // properties
    context_type        _context_type;
    unsigned            _refresh_rate;

    device_context_config();
    //device_context_config(const device_context_config& fmt);

    //device_context_config&   operator=(const device_context_config& rhs);
    //void                    swap(device_context_config& fmt);

    //bool                    operator==(const device_context_config& fmt) const;
    //bool                    operator!=(const device_context_config& fmt) const;

}; // struct device_context_config

typedef scm::shared_ptr<device>     device_ptr;

// the starting point, without a device nothing will happen
device_ptr __scm_export(ogl)
create_device(const device_initializer& init,
              const device_context_config& cfg);

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_SCMGL_SCMGL_H_INCLUDED
