
#ifndef VOLUME_RENDERER_PARAMETERS_H_INCLUDED
#define VOLUME_RENDERER_PARAMETERS_H_INCLUDED

// includes, project
#include <scm_core/math/math.h>
#include <ogl/textures/texture_1d.h>
#include <ogl/textures/texture_3d.h>

namespace gl
{
    class texture_3d;

    class volume_renderer_parameters
    {
    public:
        volume_renderer_parameters();
        volume_renderer_parameters(const gl::volume_renderer_parameters& prop);
        virtual ~volume_renderer_parameters();

        const volume_renderer_parameters& operator=(const gl::volume_renderer_parameters& rhs);

        math::vec3f_t           _point_of_interest;
        math::vec3f_t           _extend;
        math::vec3f_t           _aspect;
        math::vec3f_t           _voxel_size;
        math::vec3f_t           _cross_plane_positions;
        
        math::vec2f_t           _screen_dimensions;

        float                   _step_size;

        gl::texture_1d          _color_alpha_texture;
        gl::texture_3d          _volume_texture;

        unsigned int            _geom_depth_texture_id;

    private:
    }; // class volume_renderer_parameters

} // namespace gl

#endif // VOLUME_RENDERER_PARAMETERS_H_INCLUDED
