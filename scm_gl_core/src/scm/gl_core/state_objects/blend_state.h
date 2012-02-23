
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_BLEND_STATE_H_INCLUDED
#define SCM_GL_CORE_BLEND_STATE_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device/device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class blend_ops_array;

class __scm_export(gl_core) blend_ops
{
public:
    blend_ops(bool            in_enabled,
              blend_func      in_src_rgb_func    = FUNC_ONE,
              blend_func      in_dst_rgb_func    = FUNC_ZERO,
              blend_func      in_src_alpha_func  = FUNC_ONE,
              blend_func      in_dst_alpha_func  = FUNC_ZERO,
              blend_equation  in_rgb_equation    = EQ_FUNC_ADD,
              blend_equation  in_alpha_equation  = EQ_FUNC_ADD,
              unsigned        in_write_mask      = COLOR_ALL);

    blend_ops_array      operator()(bool            in_enabled,
                                    blend_func      in_src_rgb_func    = FUNC_ONE,
                                    blend_func      in_dst_rgb_func    = FUNC_ZERO,
                                    blend_func      in_src_alpha_func  = FUNC_ONE,
                                    blend_func      in_dst_alpha_func  = FUNC_ZERO,
                                    blend_equation  in_rgb_equation    = EQ_FUNC_ADD,
                                    blend_equation  in_alpha_equation  = EQ_FUNC_ADD,
                                    unsigned        in_write_mask      = COLOR_ALL);

    bool operator==(const blend_ops& rhs) const;
    bool operator!=(const blend_ops& rhs) const;

public:
    bool            _enabled;
    blend_func      _src_rgb_func;
    blend_func      _dst_rgb_func;
    blend_equation  _rgb_equation;
    blend_func      _src_alpha_func;
    blend_func      _dst_alpha_func;
    blend_equation  _alpha_equation;
    unsigned        _write_mask;

}; // class blend_state_desc

class __scm_export(gl_core) blend_ops_array
{
public:
    typedef std::vector<blend_ops> blend_ops_vector;

public:
    blend_ops_array(const blend_ops_array& in_blend_op_array);
    blend_ops_array(const blend_ops& in_blend_ops);
    blend_ops_array(bool            in_enabled,
                    blend_func      in_src_rgb_func    = FUNC_ONE,
                    blend_func      in_dst_rgb_func    = FUNC_ZERO,
                    blend_func      in_src_alpha_func  = FUNC_ONE,
                    blend_func      in_dst_alpha_func  = FUNC_ZERO,
                    blend_equation  in_rgb_equation    = EQ_FUNC_ADD,
                    blend_equation  in_alpha_equation  = EQ_FUNC_ADD,
                    unsigned        in_write_mask      = COLOR_ALL);

    blend_ops_array& operator()(const blend_ops& in_blend_ops);
    blend_ops_array& operator()(bool            in_enabled,
                                blend_func      in_src_rgb_func    = FUNC_ONE,
                                blend_func      in_dst_rgb_func    = FUNC_ZERO,
                                blend_func      in_src_alpha_func  = FUNC_ONE,
                                blend_func      in_dst_alpha_func  = FUNC_ZERO,
                                blend_equation  in_rgb_equation    = EQ_FUNC_ADD,
                                blend_equation  in_alpha_equation  = EQ_FUNC_ADD,
                                unsigned        in_write_mask      = COLOR_ALL);

    const blend_ops& operator[](int in_index) const;

    size_t                  size() const;
    const blend_ops_vector& blend_operations() const;

    bool operator==(const blend_ops_array& rhs) const;
    bool operator!=(const blend_ops_array& rhs) const;

private:
    blend_ops_vector     _array;

}; // class blend_ops_array


struct __scm_export(gl_core) blend_state_desc {
    blend_state_desc(const blend_ops& in_blend_ops = blend_ops(false), bool in_alpha_to_coverage = false);
    blend_state_desc(const blend_ops_array& in_blend_ops, bool in_alpha_to_coverage = false);

    blend_ops_array         _blend_ops;
    bool                    _alpha_to_coverage;
}; // struct blend_state_desc

class __scm_export(gl_core) blend_state : public render_device_child
{
public:
    virtual ~blend_state();

    const blend_state_desc& descriptor() const;

protected:
    blend_state(      render_device&    in_device,
                const blend_state_desc& in_desc);

    void                apply(const render_context& in_context, const math::vec4f& in_blend_color,
                              const blend_state&    in_applied_state, const math::vec4f& in_applied_blend_color) const;
    void                force_apply(const render_context& in_context, const math::vec4f& in_blend_color) const;


    void                force_apply(const render_context& in_context,
                                    const blend_ops&      in_blend_ops) const;
    void                force_apply_i(const render_context& in_context,
                                      unsigned              in_index,
                                      const blend_ops&      in_blend_ops) const;
    void                checked_apply(const render_context& in_context,
                                      const blend_ops&      in_blend_ops,
                                      const blend_ops&      in_applied_blend_ops) const;
    void                checked_apply_i(const render_context& in_context,
                                        unsigned              in_index,
                                        const blend_ops&      in_blend_ops,
                                        const blend_ops&      in_applied_blend_ops) const;
    void                force_disable_i(const render_context& in_context,
                                        unsigned              in_index) const;

protected:
    blend_state_desc    _descriptor;

private:
    friend class scm::gl::render_device;
    friend class scm::gl::render_context;
}; // class blend_state

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_BLEND_STATE_H_INCLUDED
