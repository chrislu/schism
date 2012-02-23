
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_DEPTH_STENCIL_STATE_H_INCLUDED
#define SCM_GL_CORE_DEPTH_STENCIL_STATE_H_INCLUDED

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) stencil_ops {
    stencil_ops(compare_func in_stencil_func  = COMPARISON_ALWAYS,
                stencil_op   in_stencil_sfail = STENCIL_KEEP,
                stencil_op   in_stencil_dfail = STENCIL_KEEP,
                stencil_op   in_stencil_dpass = STENCIL_KEEP);

    bool operator==(const stencil_ops& rhs) const;
    bool operator!=(const stencil_ops& rhs) const;

    compare_func    _stencil_func;
    stencil_op      _stencil_sfail;
    stencil_op      _stencil_dfail;
    stencil_op      _stencil_dpass;
};

struct __scm_export(gl_core) depth_stencil_state_desc {
    depth_stencil_state_desc(bool in_depth_test = true, bool in_depth_mask = true, compare_func in_depth_func = COMPARISON_LESS,
                             bool in_stencil_test = false, unsigned in_stencil_rmask = ~0u, unsigned in_stencil_wmask = ~0u,
                             const stencil_ops& in_stencil_ops = stencil_ops());
    depth_stencil_state_desc(bool in_depth_test, bool in_depth_mask, compare_func in_depth_func,
                             bool in_stencil_test, unsigned in_stencil_rmask, unsigned in_stencil_wmask,
                             const stencil_ops& in_stencil_front_ops, const stencil_ops& in_stencil_back_ops);

    bool            _depth_test;
    bool            _depth_mask;
    compare_func    _depth_func;

    bool            _stencil_test;
    unsigned        _stencil_rmask;
    unsigned        _stencil_wmask;
    stencil_ops     _stencil_front_ops;
    stencil_ops     _stencil_back_ops;
}; // struct depth_stencil_state_desc

class __scm_export(gl_core) depth_stencil_state : public render_device_child
{
public:
    virtual ~depth_stencil_state();

    const depth_stencil_state_desc& descriptor() const;
protected:
    depth_stencil_state(render_device&                  in_device,
                        const depth_stencil_state_desc& in_desc);

    void                            apply(const render_context& in_context, unsigned in_stencil_ref,
                                          const depth_stencil_state& in_applied_state, unsigned in_applied_stencil_ref) const;
    void                            force_apply(const render_context& in_context, unsigned in_stencil_ref) const;

protected:
    depth_stencil_state_desc        _descriptor;

private:
    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class depth_stencil_state

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_DEPTH_STENCIL_STATE_H_INCLUDED
