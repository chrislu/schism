
#ifndef SCM_GL_CORE_VIEWPORT_H_INCLUDED
#define SCM_GL_CORE_VIEWPORT_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) viewport
{
public:
    explicit viewport(const math::vec2ui& in_lower_left,
                      const math::vec2ui& in_dimensions,
                      const math::vec2f&  in_depth_range = math::vec2f(0.0f, 1.0f));

    bool operator==(const viewport& rhs) const;
    bool operator!=(const viewport& rhs) const;

    math::vec2ui    _lower_left;
    math::vec2ui    _dimensions;
    math::vec2f     _depth_range;

}; // class viewport

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_VIEWPORT_H_INCLUDED
