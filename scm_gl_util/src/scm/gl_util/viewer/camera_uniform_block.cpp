
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "camera_uniform_block.h"

#include <string>

#include <scm/log.h>

#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>

#include <scm/gl_util/viewer/camera.h>

namespace {

const std::string camera_block_include_path = "/scm/gl_util/camera_block.glslh";
const std::string camera_block_include_src  = "     \
    #ifndef SCM_GL_UTIL_CAMERA_BLOCK_INCLUDED       \n\
    #define SCM_GL_UTIL_CAMERA_BLOCK_INCLUDED       \n\
                                                    \n\
    layout(std140, column_major)                    \n\
    uniform camera_matrices                         \n\
    {                                               \n\
        vec4 ws_position;                           \n\
        vec4 ws_near_plane;                         \n\
                                                    \n\
        mat4 v_matrix;                              \n\
        mat4 v_matrix_inverse;                      \n\
        mat4 v_matrix_inverse_transpose;            \n\
                                                    \n\
        mat4 p_matrix;                              \n\
        mat4 p_matrix_inverse;                      \n\
                                                    \n\
        mat4 vp_matrix;                             \n\
        mat4 vp_matrix_inverse;                     \n\
    } camera_transform;                             \n\
                                                    \n\
    #endif // SCM_GL_UTIL_CAMERA_BLOCK_INCLUDED     \n\
                                                    \n\
    ";

} // namespace 

namespace scm {
namespace gl {

camera_uniform_block::camera_uniform_block(const render_device_ptr& device)
{
    _uniform_block = make_uniform_block<camera_block>(device);
    add_block_include_string(device);
}

camera_uniform_block::~camera_uniform_block()
{
    _uniform_block.reset();
}

void
camera_uniform_block::update(const render_context_ptr& context,
                             const camera&             cam)
{
    _uniform_block.begin_manipulation(context); {
        _uniform_block->_ws_position                 = cam.position();
        _uniform_block->_ws_near_plane               = cam.view_frustum().get_plane(frustum::near_plane).vector();
        _uniform_block->_p_matrix                    = cam.projection_matrix();
        _uniform_block->_p_matrix_inverse            = cam.projection_matrix_inverse();
        _uniform_block->_v_matrix                    = cam.view_matrix();
        _uniform_block->_v_matrix_inverse            = cam.view_matrix_inverse();
        _uniform_block->_v_matrix_inverse_transpose  = cam.view_matrix_inverse_transpose();
        _uniform_block->_vp_matrix                   = cam.view_projection_matrix();
        _uniform_block->_vp_matrix_inverse           = cam.view_projection_matrix_inverse();
    } _uniform_block.end_manipulation();
}

const camera_uniform_block::block_type&
camera_uniform_block::block() const
{
    return _uniform_block;
}

/*static*/
void
camera_uniform_block::add_block_include_string(const render_device_ptr& device)
{
    if (!device->add_include_string(camera_block_include_path, camera_block_include_src)) {
        scm::err() << "camera_uniform_block::add_block_include_string(): error adding camera block include string." << log::end;
    }
}

} // namespace gl
} // namespace scm
