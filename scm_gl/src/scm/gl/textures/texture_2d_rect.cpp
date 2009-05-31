
#include "texture_2d_rect.h"

namespace scm {
namespace gl {

texture_2d_rect::texture_2d_rect()
: texture_2d(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_RECTANGLE_ARB)
{
}

texture_2d_rect::~texture_2d_rect()
{
}

} // namespace gl
} // namespace scm
