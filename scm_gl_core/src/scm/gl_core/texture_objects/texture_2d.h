
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_2D_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_2D_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/texture_objects/texture_image.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) texture_2d_desc
{
    texture_2d_desc(const math::vec2ui& in_size,
                    const data_format   in_format,
                    const unsigned      in_mip_levels = 1,
                    const unsigned      in_array_layers = 1,
                    const unsigned      in_samples = 1);

    bool operator==(const texture_2d_desc& rhs) const;
    bool operator!=(const texture_2d_desc& rhs) const;

    math::vec2ui    _size;
    data_format     _format;
    unsigned        _mip_levels;
    unsigned        _array_layers;
    unsigned        _samples;
    unsigned        _flags; // auto generate mipmaps
}; // struct texture_2d_desc

class __scm_export(gl_core) texture_2d : public texture_image
{
public:
    virtual ~texture_2d();

    const texture_2d_desc&  descriptor() const;
    void                    print(std::ostream& os) const {};

    data_format             format() const;
    math::vec2ui            dimensions() const;
    unsigned                array_layers() const;
    unsigned                mip_map_layers() const;
    unsigned                samples() const;

protected:
    texture_2d(render_device&            in_device,
               const texture_2d_desc&    in_desc);
    texture_2d(render_device&            in_device,
               const texture_2d_desc&    in_desc,
               const data_format         in_initial_data_format,
               const std::vector<void*>& in_initial_mip_level_data);
    texture_2d(render_device&            in_device,
               const texture_2d&         in_orig_texture,
               const data_format         in_data_format,
               const math::vec2ui&       in_mip_range,
               const math::vec2ui&       in_layer_range);

    bool                    allocate_storage(const render_device&      in_device,
                                             const texture_2d_desc&    in_desc);
    bool                    upload_initial_data(const render_device&      in_device,
                                                const texture_2d_desc&    in_desc,
                                                const data_format         in_initial_data_format,
                                                const std::vector<void*>& in_initial_mip_level_data);
    bool                    image_sub_data(const render_context& in_context,
                                           const texture_region& in_region,
                                           const unsigned        in_level,
                                           const data_format     in_data_format,
                                           const void*const      in_data);
    bool                    create_texture_view(const render_device&      in_device,
                                                const texture_2d&         in_orig_texture,
                                                const data_format         in_data_format,
                                                const math::vec2ui&       in_mip_range,
                                                const math::vec2ui&       in_layer_range);

protected:
    texture_2d_desc         _descriptor;

private:
    friend class render_device;
    friend class render_context;

}; // class texture_2d

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_2D_H_INCLUDED
