
#ifndef SCM_GL_CORE_TRANSFORM_FEEDBACK_H_INCLUDED
#define SCM_GL_CORE_TRANSFORM_FEEDBACK_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/context_bindable_object.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) transform_feedback : 
#if SCM_GL_CORE_BASE_OPENGL_VERSION >= SCM_GL_CORE_OPENGL_VERSION_400
    public context_bindable_object, 
#endif
    public render_device_child
{
}; // class transform_feedback

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TRANSFORM_FEEDBACK_H_INCLUDED
