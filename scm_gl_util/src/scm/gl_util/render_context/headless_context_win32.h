
#ifndef SCM_GL_UTIL_HEADLESS_CONTEXT_WIN32_H_INCLUDED
#define SCM_GL_UTIL_HEADLESS_CONTEXT_WIN32_H_INCLUDED

#include <scm/core/pointer_types.h>

#include <scm/gl_util/render_context/detail/wgl.h>
#include <scm/gl_util/render_context/headless_context.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) headless_context_win32 : public headless_context
{
public:
    headless_context_win32();
    virtual ~headless_context_win32();

    bool            setup(const context_format& desc,
                          const window_context& partent_ctx);
    void            cleanup();

    bool            make_current(bool current = true) const;

protected:
    handle                  _pbuffer;
    shared_ptr<detail::wgl> _wgl;

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_HEADLESS_CONTEXT_WIN32_H_INCLUDED
