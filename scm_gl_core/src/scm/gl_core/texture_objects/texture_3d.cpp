
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_3d.h"

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

texture_3d_desc::texture_3d_desc(const math::vec3ui& in_size,
                                 const data_format   in_format,
                                 const unsigned      in_mip_levels)
  : _size(in_size)
  , _format(in_format)
  , _mip_levels(in_mip_levels)
{
}

bool
texture_3d_desc::operator==(const texture_3d_desc& rhs) const
{
    return (   (_size         == rhs._size)
            && (_format       == rhs._format)
            && (_mip_levels   == rhs._mip_levels));
}

bool
texture_3d_desc::operator!=(const texture_3d_desc& rhs) const
{
    return (   (_size         != rhs._size)
            || (_format       != rhs._format)
            || (_mip_levels   != rhs._mip_levels));
}

texture_3d::texture_3d(render_device&           in_device,
                       const texture_3d_desc&   in_desc)
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

    gl_assert(glapi, leaving texture_3d::texture_3d());
}

texture_3d::texture_3d(render_device&            in_device,
                       const texture_3d_desc&    in_desc,
                       const data_format         in_initial_data_format,
                       const std::vector<void*>& in_initial_mip_level_data)
  : texture_image(in_device)
  , _descriptor(in_desc)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    
    //if (state().ok()) {
    //    image_data(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data);
    //}
    if (state().ok()) {
        allocate_storage(in_device, in_desc);
    }
    if (state().ok()) {
        if (in_initial_mip_level_data.size() > 0) {
            upload_initial_data(in_device, in_desc, in_initial_data_format, in_initial_mip_level_data);
        }
    }

    gl_assert(glapi, leaving texture_3d::texture_3d());
}

texture_3d::texture_3d(render_device&            in_device,
                       const texture_3d&         in_orig_texture,
                       const data_format         in_data_format,
                       const math::vec2ui&       in_mip_range)
  : texture_image(in_device)
  , _descriptor(in_orig_texture.descriptor())
{
    const opengl::gl_core& glapi = in_device.opengl_api();

    if (SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_430) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
    }
    else {
        if (state().ok()) {
            if(in_orig_texture.state().ok()) {
                create_texture_view(in_device, in_orig_texture, in_data_format, in_mip_range);
            }
            else {
                state().set(object_state::OS_ERROR_INVALID_VALUE);
            }
        }
    }
    gl_assert(glapi, leaving texture_3d::texture_2d());
}

texture_3d::~texture_3d()
{
}

const texture_3d_desc&
texture_3d::descriptor() const
{
    return _descriptor;
}

bool
texture_3d::allocate_storage(const render_device&      in_device,
                             const texture_3d_desc&    in_desc)
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

    context_bindable_object::_gl_object_target  = GL_TEXTURE_3D;
    context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_3D;

#if !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
    glapi.glBindTexture(object_target(), object_id());
#endif // !SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS
    if (SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_420) {//false) { //BUG r280 
        //glerr() << "storage" << log::end;
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureStorage3DEXT(object_id(), object_target(), init_mip_levels, gl_internal_format, in_desc._size.x, in_desc._size.y, in_desc._size.z); 
        }
        else { // no EXT_DIRECT_STATE_ACCESS
            glapi.glTexStorage3D(object_target(), init_mip_levels, gl_internal_format, in_desc._size.x, in_desc._size.y, in_desc._size.z);
        }
        gl_assert(glapi, texture_3d::image_data() after glTexStorage3D());
    }
    else { // SCM_GL_CORE_OPENGL_CORE_VERSION < SCM_GL_CORE_OPENGL_CORE_VERSION_420
        // make sure the unpack buffer is not bound!
        util::buffer_binding_guard upbg(glapi, GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING);
        glapi.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        for (unsigned i = 0; i < init_mip_levels; ++i) {
            math::vec3ui lev_size = util::mip_level_dimensions(in_desc._size, i);
            if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
                glapi.glTextureImage3DEXT(object_id(), object_target(), i, gl_internal_format, lev_size.x, lev_size.y, lev_size.z, 0, gl_base_format, gl_base_type, 0);
            }
            else {
                glapi.glTexImage3D(object_target(), i, gl_internal_format, lev_size.x, lev_size.y, lev_size.z, 0, gl_base_format, gl_base_type, 0);
            }

            gl_assert(glapi, texture_3d::image_data() after glTexImage3D());
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
texture_3d::upload_initial_data(const render_device&      in_device,
                                const texture_3d_desc&    in_desc,
                                const data_format         in_initial_data_format,
                                const std::vector<void*>& in_initial_mip_level_data)
{
    using namespace scm::math;

    for (int l = 0, s = static_cast<int>(in_initial_mip_level_data.size()); l < s; ++l) {
        const vec3ui         ls = util::mip_level_dimensions(in_desc._size, l);
        const texture_region r(vec3ui(0u), ls);
        const void*          d = in_initial_mip_level_data[l];
        if (!image_sub_data(*(in_device.main_context()), r, l, in_initial_data_format, d)) {
            return false;
        }
    }

    return true;
}

bool
texture_3d::image_sub_data(const render_context& in_context,
                           const texture_region& in_region,
                           const unsigned        in_level,
                           const data_format     in_data_format,
                           const void*const      in_data)
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
                << "texture_3d::image_sub_data(): currently not supporting compressed data formats" << log::end;
        return false;

#if 0
        scm::size_t w          = (in_region._dimensions.x + 3) / 4;
        scm::size_t h          = (in_region._dimensions.y + 3) / 4;
        scm::size_t d          =  in_region._dimensions.z;
        scm::size_t image_size = w * h * d * compressed_block_size(in_data_format);

        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glCompressedTextureSubImage3DEXT(object_id(), object_target(),
                                            in_level,
                                            in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                            in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                            gl_internal_format,
                                            static_cast<int>(image_size),
                                            in_data);
        }
        else {
            util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
            glapi.glBindTexture(object_target(), object_id());

            glapi.glCompressedTexSubImage3D(object_target(),
                                    in_level,
                                    in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                    in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                    gl_internal_format,
                                    static_cast<int>(image_size),
                                    in_data);
        }
        gl_assert(glapi, texture_2d::image_sub_data() after glCompressedTexSubImage3D());
#endif
    }
    else {
        if (SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS) {
            glapi.glTextureSubImage3DEXT(object_id(), object_target(),
                                            in_level,
                                            in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                            in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                            gl_base_format,
                                            gl_base_type,
                                            in_data);
        }
        else {
            util::texture_binding_guard save_guard(glapi, object_target(), object_binding());
            glapi.glBindTexture(object_target(), object_id());

            glapi.glTexSubImage3D(object_target(),
                                    in_level,
                                    in_region._origin.x,     in_region._origin.y,     in_region._origin.z,
                                    in_region._dimensions.x, in_region._dimensions.y, in_region._dimensions.z,
                                    gl_base_format,
                                    gl_base_type,
                                    in_data);
        }
        gl_assert(glapi, texture_2d::image_sub_data() after glTexSubImage3D());
    }

    return true;
}

bool
texture_3d::create_texture_view(const render_device&      in_device,
                                const texture_3d&         in_orig_texture,
                                const data_format         in_data_format,
                                const math::vec2ui&       in_mip_range)
{
    assert(state().ok());

    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error         glerror(glapi);
    
    if (in_mip_range.y <= in_mip_range.x) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    const unsigned nb_mip_levels      =   math::min(_descriptor._mip_levels, in_mip_range.y)
                                        - math::min(_descriptor._mip_levels, in_mip_range.x);

    if (nb_mip_levels < 1u) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return false;
    }

    const unsigned gl_internal_format = util::gl_internal_format(in_data_format);

    context_bindable_object::_gl_object_target  = GL_TEXTURE_3D;
    context_bindable_object::_gl_object_binding = GL_TEXTURE_BINDING_3D;

    glapi.glTextureView(object_id(),
                        object_target(),
                        in_orig_texture.object_id(),
                        gl_internal_format,
                        in_mip_range.x,
                        nb_mip_levels,
                        0,
                        1);

    gl_assert(glapi, texture_3d::create_texture_view() after glTextureView());

    if (glerror) {
        state().set(glerror.to_object_state());

        return false;
    }
    else {
        // setup view descriptor
        _descriptor._format       = in_data_format;
        _descriptor._mip_levels   = nb_mip_levels;

        return true;
    }
}

data_format
texture_3d::format() const
{
    return _descriptor._format;
}

math::vec2ui
texture_3d::dimensions() const
{
    return math::vec2ui(_descriptor._size);
}

unsigned
texture_3d::array_layers() const
{
    return _descriptor._size.z;
}

unsigned
texture_3d::mip_map_layers() const
{
    return _descriptor._mip_levels;
}

unsigned
texture_3d::samples() const
{
    return 1;
}
} // namespace gl
} // namespace scm
