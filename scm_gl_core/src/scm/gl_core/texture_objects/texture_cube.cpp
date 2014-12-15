
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_cube.h"

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>

namespace scm {
namespace gl {

texture_cube_desc::texture_cube_desc(const math::vec2ui& in_size,
                                     const data_format   in_format,
                                     const unsigned      in_mip_levels)
  : _size(in_size)
  , _format(in_format)
  , _mip_levels(in_mip_levels)
{
}

bool
texture_cube_desc::operator==(const texture_cube_desc& rhs) const
{
    return (   (_size         == rhs._size)
            && (_format       == rhs._format)
            && (_mip_levels   == rhs._mip_levels));
}

bool
texture_cube_desc::operator!=(const texture_cube_desc& rhs) const
{
    return (   (_size         != rhs._size)
            || (_format       != rhs._format)
            || (_mip_levels   != rhs._mip_levels));
}

texture_cube::texture_cube(render_device&           in_device,
                           const texture_cube_desc& in_desc)
  : texture_image(in_device)
  , _descriptor(in_desc)
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    //if (state().ok()) {
    //    image_data(in_device, in_desc, FORMAT_NULL, std::vector<void*>());
    //}
    if (state().ok()) {
        allocate_storage(in_device, in_desc);
    }

    gl_assert(glapi, leaving texture_cube::texture_cube());
}

texture_cube::texture_cube(render_device&            in_device,
                           const texture_cube_desc&  in_desc,
                           const data_format         in_initial_data_format,
                           const std::vector<void*>& in_initial_mip_level_data_px,
                           const std::vector<void*>& in_initial_mip_level_data_nx,
                           const std::vector<void*>& in_initial_mip_level_data_py,
                           const std::vector<void*>& in_initial_mip_level_data_ny,
                           const std::vector<void*>& in_initial_mip_level_data_pz,
                           const std::vector<void*>& in_initial_mip_level_data_nz)
  : texture_image(in_device)
  , _descriptor(in_desc)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    
    if (state().ok()) {
        allocate_storage(in_device, in_desc);
    }
    if (state().ok()) {
        if (in_initial_mip_level_data_px.size() > 0 &&
            in_initial_mip_level_data_nx.size() > 0 &&
            in_initial_mip_level_data_py.size() > 0 &&
            in_initial_mip_level_data_ny.size() > 0 &&
            in_initial_mip_level_data_pz.size() > 0 &&
            in_initial_mip_level_data_nz.size() > 0) {

            upload_initial_data(in_device, in_desc, in_initial_data_format, 
                                in_initial_mip_level_data_px,
                                in_initial_mip_level_data_nx,
                                in_initial_mip_level_data_py,
                                in_initial_mip_level_data_ny,
                                in_initial_mip_level_data_pz,
                                in_initial_mip_level_data_nz);
        }
    }

    gl_assert(glapi, leaving texture_cube::texture_cube());
}

texture_cube::~texture_cube()
{
}

const texture_cube_desc&
texture_cube::descriptor() const
{
    return _descriptor;
}

bool
texture_cube::allocate_storage(const render_device&      in_device,
                               const texture_cube_desc&  in_desc)
{
    assert(state().ok());

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error          glerror(glapi);

    unsigned gl_internal_format = util::gl_internal_format(in_desc._format);
    unsigned gl_base_format     = util::gl_base_format(in_desc._format);
    unsigned gl_base_type       = util::gl_base_type(in_desc._format);

    unsigned init_mip_levels = in_desc._mip_levels;
    if (init_mip_levels == 0) {
        init_mip_levels = util::max_mip_levels(in_desc._size);
    }
    else if (init_mip_levels != 1) {
        // assert(init_mip_levels == util::max_mip_levels(in_desc._size));
    }

    context_bindable_object::_gl_object_target  = GL_TEXTURE_CUBE_MAP;
    context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_CUBE_MAP;

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
    glapi.glBindTexture(object_target(), object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {//false) { //BUG r280 
        //glerr() << "storage" << log::end;
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureStorage2DEXT(object_id(), object_target(), init_mip_levels, gl_internal_format, in_desc._size.x, in_desc._size.y);
        }
        else { // no EXT_DIRECT_STATE_ACCESS
            glapi.glTexStorage2D(object_target(), init_mip_levels, gl_internal_format, in_desc._size.x, in_desc._size.y);
        }
        gl_assert(glapi, texture_cube::image_data() after glTexStorage2D());
    }
    else { // SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_420
        // make sure the unpack buffer is not bound!
        util::buffer_binding_guard upbg(glapi, GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING);
        glapi.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        for (unsigned i = 0; i < init_mip_levels; ++i) {
            math::vec3ui lev_size = util::mip_level_dimensions(in_desc._size, i);
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                // glapi.glTextureImage3DEXT(object_id(), object_target(), i, gl_internal_format, lev_size.x, lev_size.y, lev_size.z, 0, gl_base_format, gl_base_type, 0);
            }
            else {
                // glapi.glTexImage3D(object_target(), i, gl_internal_format, lev_size.x, lev_size.y, lev_size.z, 0, gl_base_format, gl_base_type, 0);
            }

            gl_assert(glapi, texture_cube::image_data() after glTexImage2D());
        }
    }

    if (glerror) {
        state().set(glerror.to_object_state());
        return false;
    }
    else {
        return true;
    }
}

bool
texture_cube::upload_initial_data(const render_device&      in_device,
                                  const texture_cube_desc&  in_desc,
                                  const data_format         in_initial_data_format,
                                  const std::vector<void*>& in_initial_mip_level_data_px,
                                  const std::vector<void*>& in_initial_mip_level_data_nx,
                                  const std::vector<void*>& in_initial_mip_level_data_py,
                                  const std::vector<void*>& in_initial_mip_level_data_ny,
                                  const std::vector<void*>& in_initial_mip_level_data_pz,
                                  const std::vector<void*>& in_initial_mip_level_data_nz)
{
    using namespace scm::math;

    auto upload = [this](const render_device&      in_device,
                         const texture_cube_desc&  in_desc,
                         const data_format         in_initial_data_format,
                         const std::vector<void*>& data, unsigned target){

        for (int l = 0, s = static_cast<int>(data.size()); l < s; ++l) {
            const vec3ui         ls = util::mip_level_dimensions(in_desc._size, l);
            const texture_region r(vec3ui(0u), ls);
            const void*          d = data[l];

            if (!image_sub_data(*(in_device.main_context()), r, l, in_initial_data_format, d, target)) {
                return false;
            }
        }

        return true;
    };

    bool success(true);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_px, GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_nx, GL_TEXTURE_CUBE_MAP_NEGATIVE_X);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_py, GL_TEXTURE_CUBE_MAP_POSITIVE_Y);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_ny, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_pz, GL_TEXTURE_CUBE_MAP_POSITIVE_Z);
    success &= upload(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data_nz, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    return success;
}

bool
texture_cube::image_sub_data(const render_context& in_context,
                             const texture_region& in_region,
                             const unsigned        in_level,
                             const data_format     in_data_format,
                             const void*const      in_data,
                             const unsigned        in_target)
{
    assert(state().ok());


    const opengl::gl_core& glapi = in_context.opengl_api();
    util::gl_error         glerror(glapi);
    
    unsigned gl_internal_format = util::gl_internal_format(in_data_format);
    unsigned gl_base_format     = util::gl_base_format(in_data_format);
    unsigned gl_base_type       = util::gl_base_type(in_data_format);

    if (is_compressed_format(in_data_format)) {

        // currently no compressed format supports 3d textures...
        glerr() << log::error
                << "texture_cube::image_sub_data(): currently not supporting compressed data formats" << log::end;
        return false;
    }
    else {
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureSubImage2DEXT(object_id(), in_target,
                                        in_level,
                                        in_region._origin.x,     in_region._origin.y,
                                        in_region._dimensions.x, in_region._dimensions.y,
                                        gl_base_format,
                                        gl_base_type,
                                        in_data);
        }
        else {
            util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
            glapi.glBindTexture(object_target(), object_id());

            glapi.glTexSubImage2D(in_target,
                                in_level,
                                in_region._origin.x,     in_region._origin.y,
                                in_region._dimensions.x, in_region._dimensions.y,
                                gl_base_format,
                                gl_base_type,
                                in_data);
        }
        gl_assert(glapi, texture_2d::image_sub_data() after glTexSubImage2D());
    }

    return true;
}

bool
texture_cube::image_sub_data(const render_context& in_context,
                             const texture_region& in_region,
                             const unsigned        in_level,
                             const data_format     in_data_format,
                             const void*const      in_data)
{
   
    bool success(true);

    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_NEGATIVE_X);
    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_POSITIVE_Y);
    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y);
    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_POSITIVE_Z);
    success &= image_sub_data(in_context, in_region, in_level, in_data_format, in_data, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    return success;
}

data_format
texture_cube::format() const
{
    return _descriptor._format;
}

math::vec2ui
texture_cube::dimensions() const
{
    return math::vec2ui(_descriptor._size);
}

unsigned
texture_cube::array_layers() const
{
    return 1;
}


unsigned
texture_cube::mip_map_layers() const
{
    return _descriptor._mip_levels;
}

unsigned
texture_cube::samples() const
{
    return 1;
}
} // namespace gl
} // namespace scm
