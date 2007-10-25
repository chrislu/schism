
#include "volume_renderer_parameters.h"

namespace gl
{
    volume_renderer_parameters::volume_renderer_parameters()
        : _step_size(32.0f),
          _point_of_interest(1.0f),
          _extend(1.0f),
          _aspect(1.0f),
          _cp_pos(0.5f, 0.75f, 0.25f),
          _anim_speed(.5f),
          _last_frame_time(0.0f)

    {
    }
    
    volume_renderer_parameters::volume_renderer_parameters(const gl::volume_renderer_parameters& prop)
        : _step_size(prop._step_size),
          _point_of_interest(prop._point_of_interest),
          _extend(prop._extend),
          _aspect(prop._aspect),
          _voxel_size(prop._voxel_size),
          _screen_dimensions(prop._screen_dimensions),
          _volume_texture(prop._volume_texture),
          _color_alpha_texture(prop._color_alpha_texture),
          _geom_depth_texture_id(prop._geom_depth_texture_id),
          _cp_pos(prop._cp_pos)
    {
    }

    volume_renderer_parameters::~volume_renderer_parameters()
    {
    }

    const volume_renderer_parameters& volume_renderer_parameters::operator=(const gl::volume_renderer_parameters& rhs)
    {
        _step_size                  = rhs._step_size;
        _point_of_interest          = rhs._point_of_interest;
        _extend                     = rhs._extend;
        _aspect                     = rhs._aspect;
        _voxel_size                 = rhs._voxel_size;
        _screen_dimensions          = rhs._screen_dimensions;
        _volume_texture             = rhs._volume_texture;
        _color_alpha_texture        = rhs._color_alpha_texture;
        _geom_depth_texture_id      = rhs._geom_depth_texture_id;
        _cp_pos                     = rhs._cp_pos;

        return (*this);
    }



} // namespace gl
