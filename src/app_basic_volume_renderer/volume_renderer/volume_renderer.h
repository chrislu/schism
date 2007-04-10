
#ifndef VOLUME_RENDERER_H_INCLUDED
#define VOLUME_RENDERER_H_INCLUDED

#include <scm_core/math/math.h>

// includes, project

#include <ogl/utilities/volume_textured_unit_cube.h>
#include <ogl/utilities/volume_crossplanes.h>

namespace gl
{
    class volume_renderer_parameters;

    class volume_renderer
    {
    public:
        volume_renderer();
        virtual ~volume_renderer();

        virtual bool            initialize();
        virtual void            frame(const gl::volume_renderer_parameters&)      = 0;
        virtual bool            shutdown()   = 0;

        void                    draw_outlines(const gl::volume_renderer_parameters&);
        void                    draw_bounding_volume(const gl::volume_renderer_parameters&);

    protected:
        gl::volume_textured_unit_cube   _cube;
        gl::volume_crossplanes          _planes;

    private:
        // declared - never defined
        volume_renderer(const volume_renderer&);
        const volume_renderer& operator=(const volume_renderer&);

    }; // class volume_renderer

} // namespace gl

#endif // VOLUME_RENDERER_H_INCLUDED
