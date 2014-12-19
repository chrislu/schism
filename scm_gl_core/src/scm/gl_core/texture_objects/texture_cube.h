
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_CUBE_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_CUBE_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/texture_objects/texture_image.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) texture_cube_desc
{
    texture_cube_desc(const math::vec2ui& in_size,
                      const data_format   in_format,
                      const unsigned      in_mip_levels = 1);

    bool operator==(const texture_cube_desc& rhs) const;
    bool operator!=(const texture_cube_desc& rhs) const;

    math::vec2ui    _size;
    data_format     _format;
    unsigned        _mip_levels;
}; // struct texture_cube_desc

class __scm_export(gl_core) texture_cube : public texture_image
{
public:
    virtual ~texture_cube();

    const texture_cube_desc&  descriptor() const;
    void                    print(std::ostream& os) const {};

    data_format             format() const;
    math::vec2ui            dimensions() const;
    unsigned                array_layers() const;
    unsigned                mip_map_layers() const;
    unsigned                samples() const;

protected:
    texture_cube(render_device&            in_device,
                 const texture_cube_desc&  in_desc);
    texture_cube(render_device&            in_device,
                 const texture_cube_desc&  in_desc,
                 const data_format         in_initial_data_format,
                 const std::vector<void*>& in_initial_mip_level_data_px,
                 const std::vector<void*>& in_initial_mip_level_data_nx,
                 const std::vector<void*>& in_initial_mip_level_data_py,
                 const std::vector<void*>& in_initial_mip_level_data_ny,
                 const std::vector<void*>& in_initial_mip_level_data_pz,
                 const std::vector<void*>& in_initial_mip_level_data_nz);

    bool                    allocate_storage(const render_device&      in_device,
                                             const texture_cube_desc&  in_desc);
    bool                    upload_initial_data(const render_device&      in_device,
                                                const texture_cube_desc&  in_desc,
                                                const data_format         in_initial_data_format,
                                                const std::vector<void*>& in_initial_mip_level_data_px,
                                                const std::vector<void*>& in_initial_mip_level_data_nx,
                                                const std::vector<void*>& in_initial_mip_level_data_py,
                                                const std::vector<void*>& in_initial_mip_level_data_ny,
                                                const std::vector<void*>& in_initial_mip_level_data_pz,
                                                const std::vector<void*>& in_initial_mip_level_data_nz);
    bool                    image_sub_data(const render_context& in_context,
                                           const texture_region& in_region,
                                           const unsigned        in_level,
                                           const data_format     in_data_format,
                                           const void*const      in_data,
                                           const unsigned        in_target);
    bool                    image_sub_data(const render_context& in_context,
                                           const texture_region& in_region,
                                           const unsigned        in_level,
                                           const data_format     in_data_format,
                                           const void*const      in_data);

protected:
    texture_cube_desc         _descriptor;

private:
    friend class render_device;
    friend class render_context;

}; // class texture_cube

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_CUBE_H_INCLUDED
