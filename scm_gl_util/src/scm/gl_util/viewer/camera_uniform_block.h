
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_CAMERA_UNIFORM_BLOCK_H_INCLUDED
#define SCM_GL_UTIL_CAMERA_UNIFORM_BLOCK_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/buffer_objects/uniform_buffer_adaptor.h>

#include <scm/gl_util/viewer/viewer_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) camera_uniform_block
{
public:
    struct camera_block {
        math::vec4f _ws_position;
        math::vec4f _ws_near_plane;
        
        math::mat4f _v_matrix;
        math::mat4f _v_matrix_inverse;
        math::mat4f _v_matrix_inverse_transpose;

        math::mat4f _p_matrix;
        math::mat4f _p_matrix_inverse;

        math::mat4f _vp_matrix;
        math::mat4f _vp_matrix_inverse;
    }; // struct camera_block
    typedef uniform_block<camera_block>     block_type;

public:
    camera_uniform_block(const render_device_ptr& device);
    /*virtual*/ ~camera_uniform_block();

    void                update(const render_context_ptr& context,
                               const camera&             cam);

    const block_type&   block() const;

public:
    static void         add_block_include_string(const render_device_ptr& device);

private:
    block_type          _uniform_block;

}; // struct camera_uniform_block

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_CAMERA_UNIFORM_BLOCK_H_INCLUDED
