
#ifndef VOLUME_RENDERER_PARAMETERS_H_INCLUDED
#define VOLUME_RENDERER_PARAMETERS_H_INCLUDED

// includes, project
#include <scm/core/math/math.h>
#include <scm/ogl/textures/texture_1d.h>
#include <scm/ogl/textures/texture_3d.h>

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
        math::vec3f_t           _cp_pos;
        math::vec3f_t           _aspect;
        math::vec3f_t           _voxel_size;
        
        math::vec2f_t           _screen_dimensions;

        float                   _step_size;

        scm::gl::texture_1d     _color_alpha_texture;
        scm::gl::texture_3d     _volume_texture;
        scm::gl::texture_3d     _uncertainty_volume_texture;

        unsigned int            _geom_depth_texture_id;

        float _anim_speed;
        float _last_frame_time;

    private:
    }; // class volume_renderer_parameters

} // namespace gl

#endif // VOLUME_RENDERER_PARAMETERS_H_INCLUDED
