
#ifndef SCM_GL_OPENGL3_DEVICE_WIN32_H_INCLUDED_DEVICE_H_INCLUDED
#define SCM_GL_OPENGL3_DEVICE_WIN32_H_INCLUDED_DEVICE_H_INCLUDED


#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/pointer_types.h>

#include <scm/gl/graphics_device/scmgl.h>
#include <scm/gl/graphics_device/opengl3/device.h>

#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace detail {

class wgl;

} // namespace detail

class __scm_export(ogl) opengl_device_win32 : public opengl_device
{
public:
    virtual ~opengl_device_win32();

protected:
    opengl_device_win32(const device_initializer& init,
                        const device_context_config& cfg);

    virtual bool                setup_render_context(const device_context_config& cfg,
                                                     unsigned                     feature_level);


protected:
    scm::scoped_ptr<detail::wgl>    _wgl;
    handle                          _hDC;

private:
    friend device_ptr create_device(const device_initializer&, const device_context_config&);
}; // class device

} // namespace gl
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_OPENGL3_DEVICE_WIN32_H_INCLUDED_DEVICE_H_INCLUDED
