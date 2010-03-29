
#include <scm/gl_core/opengl/gl3_core.h>

#include <boost/unordered_set.hpp>
#include <boost/assign/list_of.hpp>

namespace scm {
namespace gl {
namespace util {

inline
data_type
from_gl_data_type(unsigned gl_type)
{
    switch (gl_type) {
        case GL_FLOAT:                  return (TYPE_FLOAT);break;
        case GL_FLOAT_VEC2:             return (TYPE_VEC2F);break;
        case GL_FLOAT_VEC3:             return (TYPE_VEC3F);break;
        case GL_FLOAT_VEC4:             return (TYPE_VEC4F);break;
                                        
        case GL_FLOAT_MAT3:             return (TYPE_MAT3F);break;
        case GL_FLOAT_MAT4:             return (TYPE_MAT4F);break;

        case  GL_FLOAT_MAT2x3:          return (TYPE_MAT2X3F);break;
        case  GL_FLOAT_MAT2x4:          return (TYPE_MAT2X4F);break;
        case  GL_FLOAT_MAT3x2:          return (TYPE_MAT3X2F);break;
        case  GL_FLOAT_MAT3x4:          return (TYPE_MAT3X4F);break;
        case  GL_FLOAT_MAT4x2:          return (TYPE_MAT4X2F);break;
        case  GL_FLOAT_MAT4x3:          return (TYPE_MAT4X3F);break;

        case GL_INT:                    return (TYPE_INT);break;
        case GL_INT_VEC2:               return (TYPE_VEC2I);break;
        case GL_INT_VEC3:               return (TYPE_VEC3I);break;
        case GL_INT_VEC4:               return (TYPE_VEC4I);break;
                                        
        case GL_UNSIGNED_INT:           return (TYPE_UINT);break;
        case GL_UNSIGNED_INT_VEC2:      return (TYPE_VEC2UI);break;
        case GL_UNSIGNED_INT_VEC3:      return (TYPE_VEC3UI);break;
        case GL_UNSIGNED_INT_VEC4:      return (TYPE_VEC4UI);break;
                                        
        case GL_BOOL:                   return (TYPE_BOOL);break;
        case GL_BOOL_VEC2:              return (TYPE_VEC2B);break;
        case GL_BOOL_VEC3:              return (TYPE_VEC3B);break;
        case GL_BOOL_VEC4:              return (TYPE_VEC4B);break;
                                        
        case GL_FLOAT_MAT2:             return (TYPE_MAT2F);break;

        default:
                                        return (TYPE_UNKNOWN);break;
    }
}

inline
bool
is_sampler_type(unsigned gl_type)
{
    static boost::unordered_set<unsigned> sampler_types
        = boost::assign::list_of
            (GL_SAMPLER_1D)
            (GL_SAMPLER_1D)
            (GL_SAMPLER_2D)
            (GL_SAMPLER_3D)
            (GL_SAMPLER_CUBE)
            (GL_SAMPLER_1D_SHADOW)
            (GL_SAMPLER_2D_SHADOW)
            (GL_SAMPLER_1D_ARRAY)
            (GL_SAMPLER_2D_ARRAY)
            (GL_SAMPLER_1D_ARRAY_SHADOW)
            (GL_SAMPLER_2D_ARRAY_SHADOW)
            (GL_SAMPLER_2D_MULTISAMPLE)
            (GL_SAMPLER_2D_MULTISAMPLE_ARRAY)
            (GL_SAMPLER_CUBE_SHADOW)
            (GL_SAMPLER_BUFFER)
            (GL_SAMPLER_2D_RECT)
            (GL_SAMPLER_2D_RECT_SHADOW)
            (GL_INT_SAMPLER_1D)
            (GL_INT_SAMPLER_2D)
            (GL_INT_SAMPLER_3D)
            (GL_INT_SAMPLER_CUBE)
            (GL_INT_SAMPLER_1D_ARRAY)
            (GL_INT_SAMPLER_2D_ARRAY)
            (GL_INT_SAMPLER_2D_MULTISAMPLE)
            (GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY)
            (GL_INT_SAMPLER_BUFFER)
            (GL_INT_SAMPLER_2D_RECT)
            (GL_UNSIGNED_INT_SAMPLER_1D)
            (GL_UNSIGNED_INT_SAMPLER_2D)
            (GL_UNSIGNED_INT_SAMPLER_3D)
            (GL_UNSIGNED_INT_SAMPLER_CUBE)
            (GL_UNSIGNED_INT_SAMPLER_1D_ARRAY)
            (GL_UNSIGNED_INT_SAMPLER_2D_ARRAY)
            (GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE)
            (GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY)
            (GL_UNSIGNED_INT_SAMPLER_BUFFER)
            (GL_UNSIGNED_INT_SAMPLER_2D_RECT);

    return (sampler_types.find(gl_type) != sampler_types.end());
}

} // namespace util
} // namespace gl
} // namespace scm
