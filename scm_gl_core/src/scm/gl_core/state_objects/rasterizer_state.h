
#ifndef SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED
#define SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) rasterizer_state_desc {
    rasterizer_state_desc(fill_mode             in_fmode = FILL_SOLID,
                          cull_mode             in_cmode = CULL_BACK,
                          polygon_orientation   in_fface = ORIENT_CCW,
                          bool                  in_msample = false,
                          bool                  in_sctest = false,
                          bool                  in_smlines = false);

    fill_mode               _fill_mode;
    cull_mode               _cull_mode;

    polygon_orientation     _front_face;

    bool                    _multi_sample;
    bool                    _scissor_test;
    bool                    _smooth_lines;
}; // struct depth_stencil_state_desc

class __scm_export(gl_core) rasterizer_state : public render_device_child
{
public:
    virtual ~rasterizer_state();

    const rasterizer_state_desc&    descriptor() const;

protected:
    rasterizer_state(      render_device&         in_device,
                     const rasterizer_state_desc& in_desc);

    void                            apply(const render_context&   in_context,
                                          const rasterizer_state& in_applied_state) const;
    void                            force_apply(const render_context&   in_context) const;

protected:
    rasterizer_state_desc           _descriptor;

private:
    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class rasterizer_state

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RASTERIZER_STATE_H_INCLUDED
