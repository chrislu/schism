
#ifndef SCM_MATH_MAT_GL_LIB_H_INCLUDED
#define SCM_MATH_MAT_GL_LIB_H_INCLUDED

#include <scm/ogl.h>

namespace math
{
    inline void get_gl_matrix(const int type, mat_gl<float>& m)
    {
        glGetFloatv(type, m.mat_array);
    }

    inline void get_gl_matrix(const int type, mat_gl<double>& m)
    {
        glGetDoublev(type, m.mat_array);
    }

    // TBD
#if 0
    inline mat4_t gl_ortho_matrix(  const float left,
                                    const float right,
                                    const float bottom,
                                    const float top,
                                    const float near_z,
                                    const float far_z)
    {
        float A,B,C,D,E,F;

	    A=-(right+left)/(right-left);
	    B=-(top+bottom)/(top-bottom);
	    C=-(far_z+near_z)/(far_z-near_z);

	    D=-2.0f/(far_z-near_z);
	    E=2.0f/(top-bottom);
	    F=2.0f/(right-left);

        return (mat4_t( F, 0, 0, 0,
                        0, E, 0, 0,
                        0, 0, D, 0,
                        A, B, C, 1));
    }

    inline mat4_t gl_frustum_matrix(const float left,
                                    const float right,
                                    const float bottom,
                                    const float top,
                                    const float near_z,
                                    const float far_z)
    {
        float A,B,C,D,E,F;

	    A=(right+left)/(right-left);
	    B=(top+bottom)/(top-bottom);
	    C=-(far_z+near_z)/(far_z-near_z);
	    D=-(2.0f*far_z*near_z)/(far_z-near_z);
	    E=2.0f*near_z/(top-bottom);
	    F=2.0f*near_z/(right-left);

	    return (mat4_t(	F, 0, 0, 0,
                        0, E, 0, 0,
                        A, B, C,-1,
                        0, 0, D, 0));
    }

    inline mat4_t gl_perspective_matrix( const float fovy   = 45,      // = 45, 
                                         const float aspect = 4.0f/3.0f,
                                         const float near_z = 0.1f,
                                         const float far_z  = 100)
    {
        float maxy = tanf( deg2rad(fovy*.5f)) * near_z;
        float maxx = maxy*aspect;

        return (gl_frustum_matrix(-maxx, maxx, -maxy, maxy, near_z, far_z));
    }
#endif

} // namespace math

#endif // SCM_MATH_MAT_GL_LIB_H_INCLUDED

