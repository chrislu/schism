
#ifndef FONT_FACE_H_INCLUDED
#define FONT_FACE_H_INCLUDED

#include <string>

#include <scm_core/math/math.h>
#include <scm_core/resource/resource_type_base.h>

namespace scm {
namespace font {

struct face_descriptor
{
    face_descriptor() : _size(0) {}
    face_descriptor(const face_descriptor& desc) : _name(desc._name), _size(desc._size) {}

    std::string         _name;
    unsigned            _size;
};

class face : public res::resource_type_base<face_descriptor>
{
public:
    face() {}
    face(const face_descriptor& desc) : res::resource_type_base<face_descriptor>(desc) {}
    virtual ~face() {}

    math::vec2i_t       _tex_lower_left;
    math::vec2i_t       _tex_upper_right;

    unsigned            _advance;
    math::vec2i_t       _bearing;

}; // struct font_face

inline bool operator<(const face_descriptor& lhs,
                      const face_descriptor& rhs)
{
    return   ((lhs._name < rhs._name)
           || (lhs._size < rhs._size));
}

} // namespace font
} // namespace scm

#endif // FONT_FACE_H_INCLUDED
